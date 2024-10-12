import torch, logging, argparse, os, time, sys
sys.path.append(os.path.dirname(__file__))

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.parallel import DataParallel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.tensorboard.writer import SummaryWriter

from lib.data_io import GenMRIFusionDataset
from models.GenMRIFusionViT_recognition_e1 import VisionTransformer3D
from lib.tools import init_weights, fetch_list_of_backup_files
from omegaconf import OmegaConf

def criterion(x1: torch.Tensor, 
              x2: torch.Tensor, 
              loss_function: str = 'BCE', 
              **kwargs) -> torch.Tensor:
    """Computes loss value based on the defined task. We have different type of 
        loss functions like BCE and CE for classification.

    Args:
        x1 (torch.Tensor): Output of the model.
        x2 (torch.Tensor): Target value.
        loss_function (str, optional): Type of criterion being applied. Defaults
        to 'BCE'.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torch.Tensor: Loss value of given tensors.
    """ 
    if loss_function == 'BCE':
        return BCEWithLogitsLoss()(x1.squeeze(), x2.squeeze())
    else:
        metric = CrossEntropyLoss(reduction='sum')
        return metric(x1, x2)

def main():
    parser = argparse.ArgumentParser(prog = 'MultimodalRecognizer framework', 
                                     description = 'MultimodalRecognizer models',
                                     epilog = 'Check ReadMe file for more information.',
                                     )
    parser.add_argument('-c', '--config', required=True, help='Path to the config file') 
    parser.add_argument('-m', '--mask', required=True, help='Path to the mask file')  
    parser.add_argument('-t', '--dataset', required=True, help='Path to pandas dataframe that keeps list of images')    
    parser.add_argument('-s', '--save_dir', required=True, help='Path to save checkpoint')  
    parser.add_argument('-l', '--log_dir', required=True, help='Path to save logfile')  
    parser.add_argument('-f', '--fine_tune', required=False, help='Path to the checkpoint in case of fine-tuning scenario')  
    args = parser.parse_args()
    
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET, format="[ %(asctime)s ]  %(levelname)s : %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    for i in range(torch.cuda.device_count()):
        logging.debug("Available processing unit ({} : {})".format(i, torch.cuda.get_device_name(i)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (os.path.exists(args.mask)):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")       
    if not (os.path.exists(args.config)):
        raise FileNotFoundError(f"Config file not found: {args.config}")  
    if not (os.path.exists(args.dataset)):
        raise FileNotFoundError(f"DataTable: file not found: {args.dataset}") 
    if not (os.path.exists(args.save_dir)):
        raise FileNotFoundError(f"Save directory does not exist, {args.save_dir}")  
    if not (os.path.exists(args.log_dir)):
        raise FileNotFoundError(f"Log directory does not exist, {args.log_dir}")  
    if args.fine_tune is not None and not (os.path.exists(args.fine_tune)):
        raise FileNotFoundError(f"Checkpoint file does not exist, {args.fine_tune}")  
    
    logging.info("Loading configuration data ...")
    conf = OmegaConf.load(args.config)
    save_flag = conf.EXPERIMENT.save_model
    experiment_tag = conf.EXPERIMENT.tag
    experiment_name = conf.EXPERIMENT.name
    model_architecture = conf.EXPERIMENT.architecture
    fine_tuning_checkpoint = os.path.abspath(args.fine_tune) if args.fine_tune is not None else None
    checkpoints_directory = os.path.join(args.save_dir, experiment_tag)
    mask_file_path = os.path.abspath(args.mask)
    dataset_file = os.path.abspath(args.dataset)
    log_directory = os.path.join(args.log_dir, experiment_name)
    logging.info("Loading subjects data")    
    main_dataset = GenMRIFusionDataset(genMRIDataset = dataset_file,
                                       mask_file = mask_file_path,
                                       **conf.DATASET)
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
        logging.info(f'Log directory created in *logdir/{experiment_name}')
    if save_flag and not os.path.exists(checkpoints_directory):
        os.mkdir(checkpoints_directory)
        logging.info(f'Save directory created in *savedir/{experiment_tag}')   
    if main_dataset.imbalanced_weights is not None:
        logging.info(f'Class weights = {main_dataset.imbalanced_weights}')
        main_dataset.imbalanced_weights = main_dataset.imbalanced_weights.to(dev, non_blocking=True)
    if save_flag:
        backup_files = fetch_list_of_backup_files(model_architecture)
        os.system(f"cp -f {os.path.join(os.path.dirname(os.path.abspath(__file__)),'config', backup_files[0])} {os.path.join(os.path.dirname(os.path.abspath(__file__)),'models', backup_files[1])} {checkpoints_directory}")
    data_pack = {}
    data_pack['train'], data_pack['val'] = random_split(main_dataset, [.8, .2], generator=torch.Generator().manual_seed(70))
    dataloaders = {x: DataLoader(data_pack[x], batch_size=int(conf.TRAIN.batch_size), shuffle=True, num_workers=int(conf.TRAIN.workers), pin_memory=True) for x in ['train', 'val']}       
    gpu_ids = list(range(torch.cuda.device_count()))
    writer = SummaryWriter(log_dir=log_directory, comment=conf.EXPERIMENT.name)
    if model_architecture == 'ViT3D':
        base_model = VisionTransformer3D(**conf.MODEL)              
    else:
        raise Exception("The architecture is not defined.")
    if fine_tuning_checkpoint is not None:
        checkpoint = torch.load(fine_tuning_checkpoint)
        base_model.load_state_dict(checkpoint['state_dict'], strict=True)
        logging.info(f"Model weights loaded for fine-tuning.")   
    else:
        base_model.apply(init_weights)
        logging.info(f"Model weights randomly initilized.")   

    if torch.cuda.is_available():
        base_model = base_model.cuda()
        if torch.cuda.device_count() > 1:
            base_model = DataParallel(base_model, device_ids = gpu_ids)
            logging.info(f"Pytorch Distributed Data Parallel activated using gpus: {gpu_ids}")        
    optimizer = torch.optim.Adam(base_model.parameters(), lr=float(conf.TRAIN.base_lr))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf.TRAIN.step_lr), gamma=float(conf.TRAIN.weight_decay))
    best_loss = float('inf')
    logging.info(f"Optimizer: Adam , Criterion: {conf.TRAIN.loss} , lr: {conf.TRAIN.base_lr} , decay: {conf.TRAIN.weight_decay}")
    num_epochs = int(conf.TRAIN.epochs)
    phase_error = {'train': 0., 'val': 0.}
    phase_accuracy = {'train': 0., 'val': 0.}
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                base_model.train() 
            else:
                base_model.eval()
            running_loss = 0.0
            running_accuracy = 0.0
            for sample in dataloaders[phase]:
                sMRI_inp = sample['sMRI'].to(dev, non_blocking=True)
                fnc_inp = sample['FNC'].to(dev, non_blocking=True)
                snp_inp = sample['SNP'].to(dev, non_blocking=True)
                label = sample['Diagnosis'].to(dev, non_blocking=True)
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad(set_to_none=True)
                    if model_architecture == 'ViT3D':
                        preds = base_model(sMRI_inp)
                        loss = criterion(preds, label, conf.TRAIN.loss)
                        accuracy_ = (preds.round() == label).float().sum()
                    else:
                        preds = base_model(sMRI_inp, fnc_inp, snp_inp)
                        loss = criterion(preds, label, conf.TRAIN.loss)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    running_accuracy += accuracy_

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(data_pack[phase])
            epoch_accuracy = 100 * running_accuracy / len(data_pack[phase])
            phase_error[phase] = epoch_loss
            phase_accuracy[phase] = epoch_accuracy
        logging.info("Epoch {}/{} - LR: {:.5f} - Train Loss: {:.10f} and Validation Loss: {:.10f} and Training Accuracy: {:.5f} and Validation Accuracy: {:.5f}".format(epoch+1, num_epochs, scheduler.get_last_lr()[0], phase_error['train'], phase_error['val'], phase_accuracy['train'], phase_accuracy['val']))
        writer.add_scalars("Loss", {'train': phase_error['train'], 'validation': phase_error['val']}, epoch)
        if phase == 'val' and save_flag and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch,
                        'state_dict': base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': phase_error,
                        'edition': conf.EXPERIMENT.edition}, 
                        os.path.join(checkpoints_directory, 'checkpoint_{}_{}.pth'.format(epoch, time.strftime("%m%d%y_%H%M%S"))))
    
    writer.flush()
    writer.close()
    logging.info("Training procedure is done!")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
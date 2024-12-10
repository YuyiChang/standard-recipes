from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from optuna.integration import PyTorchLightningPruningCallback

##### argument parser
class StandardArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        ### dev
        self.add_argument('--debug', action='store_true', default=False)

        ### model and training
        self.add_argument('--batch_size', default=64, type=int)
        self.add_argument('--max_epochs', default=20, type=int)
        self.add_argument('--resume', default=None, type=str)

        ### telemetry
        self.add_argument('--log_dir', default='./log', type=str)

        ### hardware
        self.add_argument('--gpus', default=[0], nargs='+', type=int)
        self.add_argument('--num_workers', default=4, type=int)

    def parse_args(self):
        args = super().parse_args()
        if args.gpus[0] == -1: args.gpus = -1
        return args
    

##### generic trainer
class StandardTrainer(pl.Trainer):
    def __init__(self, sargs=None, 
                 val_check_interval=0.25, 
                 chpt_monitor_loss=True, 
                 chpt_save_all_epoch=False,
                 chpt_save_all_val_epoch=False, 
                 strategy='ddp_find_unused_parameters_false',
                 devices=None,
                 max_epochs=None,
                 log_dir=None, 
                 debug=None, 
                 monitored_loss=['val_loss_epoch'], 
                 custom_cb=list(),
                 trial = None,
                 prune_monitor = None, 
                 *args, **kwargs):
        ## checkpointing behaviors
        cbs = []
        if chpt_monitor_loss:
            for l in monitored_loss:
                filename = '{epoch}-{step}-{' + l + ':.5f}'
                chpt_cb_loss = ModelCheckpoint(
                    monitor=l,
                    mode='min',
                    verbose=True,
                    auto_insert_metric_name=True,
                    save_on_train_epoch_end=False,
                    save_top_k=-1 if chpt_save_all_val_epoch else 3,
                    filename=filename
                )
                cbs.append(chpt_cb_loss)

        chpt_cb_epoch = ModelCheckpoint(
            monitor='epoch',
            mode='max',
            save_on_train_epoch_end=True,
            save_top_k=-1 if chpt_save_all_epoch else 3,
            filename='chpt-{epoch:02d}'
        )
        cbs.append(chpt_cb_epoch)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        cbs.append(lr_monitor)

        if trial is not None:
            chpt_opt_prun = PyTorchLightningPruningCallback(trial, monitor = prune_monitor)# if trial else []
            cbs.append(chpt_opt_prun)

        print('&&&&&&&&&&&&', cbs)

        # for cb in custom_cb:
        #     cbs.append(cb)
        #     print('*********', cbs) #jw addition

        super().__init__(accelerator='cuda', 
                         devices=devices if devices is not None else sargs.gpus, 
                         max_epochs=max_epochs if max_epochs is not None else sargs.max_epochs, 
                         val_check_interval=val_check_interval, 
                         default_root_dir=log_dir if log_dir is not None else sargs.log_dir, 
                         enable_progress_bar=True, 
                         callbacks=cbs, 
                         strategy=strategy,
                         fast_dev_run=debug if debug is not None else sargs.debug, 
                         log_every_n_steps=1,
                         *args, **kwargs)
        

if __name__ == '__main__':
    parser = StandardArgParser()
    args = parser.parse_args()
    print(args)
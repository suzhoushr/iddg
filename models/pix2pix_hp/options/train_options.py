from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss') 
        self.parser.add_argument('--lambda_dtzs', type=float, default=0.0, help='weight for feature matching loss')   
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')  
        self.parser.add_argument('--no_groundtruth_loss', action='store_true', help='if specified, do *not* use L1 groundtruth loss')     
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # for distll
        self.parser.add_argument('--distill_sign', action='store_true', default=False, help='control whether distillation is used.')
        self.parser.add_argument('--distill_type', type=str, default='kd_bridge', help='selects distill type [kd_bridge, kd_ka, kd_normal]') 
        self.parser.add_argument('--lambda_distill', type=float, default=1.0, help='weight for distill loss') 
        self.parser.add_argument('--teacher_ngf', type=int, default=24, help='# of gen filters in first conv layer of teacher net')
        self.parser.add_argument('--teacher_ratio_decode', type=float, default=1.5, help='deconv layer of teacher net')
        self.parser.add_argument('--restore_teacher_G_path', type=str, default='', help='load the teacher G model from the specified location')
        self.parser.add_argument('--restore_student_G_path', type=str, default='', help='load the teacher G model from the specified location')
        self.parser.add_argument('--restore_D_path', type=str, default='', help='load the D model from the specified location,teacher D is same with student D')

        self.isTrain = True

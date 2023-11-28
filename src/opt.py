import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Random seed for np and tf (-1 to avoid seeding)
    parser.add_argument('--seed', type=int, default=0,  help='random seed (default: 0)')
    #Training parametes

    parser.add_argument('--train-repeats', type=int, default=1,  help='Number of iterations over the train set per epoch (1 means once only)')
    parser.add_argument('--val-repeats', type=int, default=1,  help='Number of iterations over the val set per epoch (1 means once only)')
    parser.add_argument('--test-repeats', type=int, default=1,  help='Number of iterations over the test set per epoch (1 means once only)')

    parser.add_argument('--n-crops', type=int, default=1,  help='Number of crops for each image')

    parser.add_argument('--train-path', type=str, default='/scratch/Lenslet_RGB/',  help='Direcory with training pngs in ODP format')
    parser.add_argument('--val-path', type=str, default='/scratch/validation_lenslet/',  help='Direcory with validation pngs in ODP format')
    parser.add_argument('--test-path', type=str, default='/scratch/validation_lenslet/',  help='Direcory with test pngs in ODP format')

    # TODO we need a separate bithdepth switch for each dataset!
    # IDM nviews
    parser.add_argument('--context-size', type=int, default=64,  help='Size of the context [64, 128] (default 64x64))')
    parser.add_argument('--predictor-size', type=int, default=32,  help='Size of the predictor [32, 64] (default 32x32)')
    parser.add_argument('--bit-depth', type=int, default=8,  help='Depth of the samples, in bits per pixel (default 8)')
    parser.add_argument('--epochs', type=int, default=100,  help='Epochs to test (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,  help='Batch size (default: 64). For crop dataloaders, teh actual BS is multiplied by crops_per_image')
    parser.add_argument('--loss', type=str, default='mse',  help='Loss functionto minimize [abs|mse|ssim]')
    parser.add_argument('--lr', type=float, default=0.001,  help='Initial learning rate (default: 0.001)')
    parser.add_argument('--lr-gamma', type=float, default=0.1,  help='Learning rate decay factor (default: 0.1)')
    parser.add_argument('--lr-min', type=float, default=0.0,  help='Learning rate decay factor (default: 0.1)')
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-scheduler", default="exponentiallr", type=str, help="the lr scheduler (default: steplr)")





    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    parser.add_argument('--augment', type=int, default=1,  help='Enables augmentation on train set - to be used iff all 4 predictors are available (default: 1)')
    #parser.add_argument('--LRDecay', type=float, default=1e-6,  help='Learning rate decay factor (default: 1e-6)')
    #parser.add_argument('-f', type=str, default='',  help='Dummy arument for maintaining jupiter compatibility')
    # TODO add option to select how many epochs the trained model shall be saved to disk


    parser.add_argument('--save', default='../runs/exp', type=str, 
                        help='Output dir')
    parser.add_argument('--project-name', default='delete', type=str)
    

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--model', default='conv', type=str)
    parser.add_argument("--my-decoder", action="store_true", default=False) # unet


    

    args = parser.parse_args()
    return args


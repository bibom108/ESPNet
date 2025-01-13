import loadData as ld
import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import VisualizeGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import torch.optim.lr_scheduler
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Sachin Mehta"

def save_visualization_as_pdf(input_tensor, output_tensor, target_tensor, mean, std, save_path):

    for i in range(len(input_tensor)):
        input_image = input_tensor[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        input_image = input_image[..., ::-1]
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        print(f"Input image range after denormalization: {input_image.min()} to {input_image.max()}")
        # exit()
        # Convert masks to numpy arrays
        predicted_mask = output_tensor[i].max(0)[1].cpu().numpy()
        ground_truth_mask = target_tensor[i].cpu().numpy()

        # Create a PDF to save the visualizations
        # with PdfPages(os.path.join(save_path, f"{i}.pdf")) as pdf:
            # plt.figure(figsize=(15, 5))

            # # Input Image
            # plt.subplot(1, 3, 1)
            # plt.imshow(input_image)
            # plt.title("Input Image")
            # plt.axis("off")

            # # Predicted Mask
            # plt.subplot(1, 3, 2)
            # plt.imshow(predicted_mask, cmap="jet")
            # plt.title("Predicted Mask")
            # plt.axis("off")

            # # Ground Truth Mask
            # plt.subplot(1, 3, 3)
            # plt.imshow(ground_truth_mask, cmap="jet")
            # plt.title("Ground Truth Mask")
            # plt.axis("off")

            # plt.tight_layout()
            # pdf.savefig()  # Save the current figure to the PDF
            # plt.close()
        
        plt.figure(figsize=(15, 5))  # Adjust the figure size if needed
        # Input Image
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title("Input Image", fontsize=16)  # Larger label
        plt.axis("off")

        # Predicted Mask
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_mask, cmap="jet")
        plt.title("Predicted Mask", fontsize=16)  # Larger label
        plt.axis("off")

        # Ground Truth Mask
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth_mask, cmap="jet")
        plt.title("Ground Truth Mask", fontsize=16)  # Larger label
        plt.axis("off")

        # Save the visualization as a PNG file
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{i}.png"), dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
        plt.close()

def val(args, val_loader, model, criterion, mean, std):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        save_visualization_as_pdf(
                    input,  # First input image in the batch
                    output,  # Predicted mask for the first image
                    target,  # Ground truth mask for the first image
                    save_path="./vis_h",
                    mean=mean,
                    std=std,
                )
        exit()

        # print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def trainValidateSegmentation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''
    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    q = args.q
    p = args.p
    # load the model
    if not args.decoder:
        model = net.ESPNet_Encoder(args.classes, p=p, q=q)
        args.savedir = args.savedir + '_enc_' + str(p) + '_' + str(q) + "_" + args.id + '/'
    else:
        model = net.ESPNet(args.classes, p=p, q=q)
        args.savedir = args.savedir + '_dec_' + str(p) + '_' + str(q) + "_" + args.id + '/'

    if args.onGPU:
        model = model.cuda()
    
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.onGPU:
        weight = weight.cuda()

    criteria = CrossEntropyLoss2d(weight) #weight

    if args.onGPU:
        criteria = criteria.cuda()

    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size + 4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if args.onGPU:
        cudnn.benchmark = True

    checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pth.tar"))
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(torch.load(os.path.join(args.savedir, "model_220.pth")))
    print("=> loaded checkpoint '{}' (epoch {})"
        .format(args.resume, checkpoint['epoch']))

    lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria, data["mean"], data["std"])
    print("="*20)
    print(format(lossVal, '.4f'))
    print(format(overall_acc_val, '.4f'))
    print(format(per_class_acc_val.mean(), '.4f'))
    print(format(per_class_iu_val.mean(), '.4f'))
    print(format(mIOU_val, '.4f'))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--id', default="0", help='ID of the experiment')
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./city", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=8, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./results_enc_', help='directory to save the results')
    parser.add_argument('--visualizeNet', type=bool, default=False, help='If you want to visualize the model structure')
    parser.add_argument('--resume', type=bool, default=False, help='Use this flag to load last checkpoint for training')  #
    parser.add_argument('--classes', type=int, default=2, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=False,help='True if ESPNet. False for ESPNet-C') # False for encoder
    parser.add_argument('--pretrained', default='../pretrained/encoder/espnet_p_2_q_8.pth', help='Pretrained ESPNet-C weights. '
                                                                              'Only used when training ESPNet')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier')

    trainValidateSegmentation(parser.parse_args())


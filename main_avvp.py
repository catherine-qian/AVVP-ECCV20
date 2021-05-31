from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from nets.net_audiovisual import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
import os
import scipy.io as sio
from nets.losses import *
from utils.display import *
import sys
import datetime


def train(args, model, train_loader, optimizer, criterion, epoch, criterion2=None):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to(args.device), sample['video_s'].to(args.device), sample[
            'video_st'].to(args.device), sample['label'].type(torch.FloatTensor).to(args.device)

        optimizer.zero_grad()
        output, a_prob, v_prob, _, x1, x2 = model(audio, video, video_st)  # x1-audio features, x2-video feature

        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        # label smoothing
        a, v = 1.0, 0.9
        Pa = a * target + (1 - a) * 0.5  # Pa=target
        Pv = v * target + (1 - v) * 0.5  # Pv=0.9*target + 0.05

        # -------------------------------------
        # individual guided learning
        loss0 = criterion(a_prob, Pa) + criterion(v_prob, Pv) + criterion(output, target)  # ([16, 25])
        # loss0 = criterion(output, target)

        # criterion2 = None
        if criterion2 is not None and epoch>15:  # not empty
            loss2 = contrast([], x2, target, criterion2)
            loss = loss0 + 0.03 * loss2
        else:
            loss = loss0
            loss2 = 0 * loss

        # loss = loss2

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(audio), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, contras Loss: {:.6f}, total loss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss0.item(), loss2.item(), loss.item()))

    if sys.gettrace() is not None:
        dispP(a_prob, v_prob, output, target)


def eval(model, val_loader, set, epoch=-1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']  # 25 classes
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')  # audio label
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')  # visual label

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            # test on each 10-sec video
            audio, video, video_st, target = sample['audio'].to(device), sample['video_s'].to(device), sample[
                'video_st'].to(device), sample['label'].to(device)  # sample
            output, a_prob, v_prob, frame_prob, *_ = model(audio, video, video_st)  # ([1, 25])
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)  # ([1, 25]) binary label

            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()  # (10, 25)  audio frame-level probability
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()  # (10, 25)

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)  # 取交集  (10, 25)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)

            # extract audio GT labels ----------------------
            GT_a = np.zeros((25, 10))  # (25, 10)
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)  # num of class label
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels --------------------------
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v  # (25, 10), (25, 10) audio/video may have different frame-level labels

            # obtain prediction matrices
            SO_a = np.transpose(Pa)  # (25, 10)
            SO_v = np.transpose(Pv)  # (25, 10)
            SO_av = SO_a * SO_v  # (25, 10)

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    segA, segV, segAV = 100 * np.mean(np.array(F_seg_a)), 100 * np.mean(np.array(F_seg_v)), 100 * np.mean(
        np.array(F_seg_av))
    print('Audio Event Detection Segment-level F1: {:.1f}'.format(segA))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(segV))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(segAV))

    avg_type = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(
        np.array(F_seg_v))) / 3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    eveA, eveV, eveAV = 100 * np.mean(np.array(F_event_a)), 100 * np.mean(np.array(F_event_v)), 100 * np.mean(
        np.array(F_event_av))
    print('Audio Event Detection Event-level F1: {:.1f}'.format(eveA))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(eveV))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(eveAV))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))

    total = segA + segV + segAV + avg_type + avg_event + eveA + eveV + eveAV + avg_type_event + avg_event_level
    print('overall ep-{}: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f} |'
          ' {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, | sum {:.1f}'.format(epoch,
                                                                         segA, segV, segAV, avg_type, avg_event,
                                                                         eveA, eveV, eveAV, avg_type_event,
                                                                         avg_event_level,
                                                                         total))
    if sys.gettrace() is not None:
        disptest(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)  # display evaluation result

    # return avg_type
    return total


def main():
    modelname = str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    print(modelname)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument(
        "--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='MMIL_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1000, metavar='S',
                        help='random seed (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='MMIL_Net',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    args.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else 'cpu')
    print(args)  # print variables

    if args.model == 'MMIL_Net':
        model = MMIL_Net().to(args.device)
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([
                ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                  st_dir=args.st_dir, transform=transforms.Compose([
                ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        # optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # args.lr=0.0003
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  #
        criterion = nn.BCELoss()
        criterion2 = SupConLoss(temperature=0.2)

        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch, criterion2=criterion2)  # train
            scheduler.step(epoch)

            F = eval(model, val_loader, args.label_val, epoch)  # validation set
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + modelname + ".pt")

        # best model-> print test result
        print("finished: best model----testing result")
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(
            torch.load(args.model_save_dir + args.checkpoint + modelname + ".pt"))  # models/MMIL_Net.pt
        eval(model, test_loader, args.label_test)

    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + modelname + ".pt"))
        eval(model, test_loader, args.label_val)
    else:
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + modelname+".pt"))  # models/MMIL_Net.pt
        eval(model, test_loader, args.label_test)


if __name__ == '__main__':
    if sys.gettrace() is not None:
        print("in debug mode")
    main()

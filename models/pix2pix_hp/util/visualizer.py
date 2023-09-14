import numpy as np
import os
import ntpath
from . import util
from . import html
import pdb

class Visualizer():
    def __init__(self, ckpts_dir, name='SHOW', display_winsize=256):
        self.win_size = display_winsize
        self.name = name

        self.web_dir = os.path.join(ckpts_dir, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        for label, image_numpy in visuals.items():
            # pdb.set_trace()
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                    util.save_image(image_numpy[i], img_path)
            else:
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

        # update website
        webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
        for n in range(epoch, 0, -1):
            webpage.add_header('epoch [%d]' % n)
            ims = []
            txts = []
            links = []

            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = 'epoch%.3d_%s_%d.png' % (n, label, i)
                        ims.append(img_path)
                        txts.append(label+str(i))
                        links.append(img_path)
                else:
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
            if len(ims) < 10:
                webpage.add_images(ims, txts, links, width=self.win_size)
            else:
                num = int(round(len(ims)/2.0))
                webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
        webpage.save()

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

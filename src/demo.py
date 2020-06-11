from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os
import os.path as osp
from lib.opts import opts  # import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from track import eval_seq

logger.setLevel(logging.INFO)


def run_demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    try:  # 视频推断的入口函数
        eval_seq(opt=opt,
                 dataloader=dataloader,
                 data_type='mot',
                 result_filename=result_filename,
                 save_dir=frame_dir,
                 show_image=False,
                 frame_rate=frame_rate)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(osp.join(result_root, 'frame'),
                    output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    run_demo(opt)

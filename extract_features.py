import os, sys, cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from os import path, mkdir
import argparse
import keyNet.utils.tools as aux
from skimage.transform import pyramid_gaussian
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
from keyNet.model.keynet_architecture import *
import keyNet.utils.desc_aux_function as loss_desc
from keyNet.model.hardnet_pytorch import *
from keyNet.datasets.dataset_utils import read_bw_image
import torch

from SOSNet import sosnet_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

confs = {
    "hardnet": {
        "name": "HardNet",
        "pytorch_weight": "keyNet/pretrained_nets/HardNet++.pth",
    },
    "sosnet": {
        "name": "SOSNet",
        "pytorch_weight": "SOSNet/sosnet-weights/sosnet-32x32-liberty.pth",
    },
}


def check_directory(dir):
    if not path.isdir(dir):
        mkdir(dir)


def create_result_dir(path):
    directories = path.split("/")
    tmp = ""
    for idx, dir in enumerate(directories):
        tmp += dir + "/"
        if idx == len(directories) - 1:
            continue
        check_directory(tmp)


def main(
    conf,
    list_images,  # File containing the image paths for extracting features.
    results_dir="extracted_features/",  # The output path to save the extracted keypoint.
    network_version="KeyNet_",  # The Key.Net network version name
    checkpoint_det_dir="keyNet/pretrained_nets/KeyNet_default",  # The path to the checkpoint file to load the detector weights.
    num_filters=8,  # The number of filters in each learnable block.
    num_learnable_blocks=3,  # The number of learnable blocks after handcrafted block.
    num_levels_within_net=3,  # The number of pyramid levels inside the architecture.
    factor_scaling_pyramid=1.2,  # The scale factor between the multi-scale pyramid levels in the architecture.
    conv_kernel_size=5,  # The size of the convolutional filters in each of the learnable blocks.
    # Multi-Scale Extractor Settings
    extract_MS=True,  # Set to True if you want to extract multi-scale features.
    num_points=1500,  # The number of desired features to extract.
    nms_size=15,  # The NMS size for computing the validation repeatability.
    border_size=15,  # The number of pixels to remove from the borders to compute the repeatability.
    order_coord="xysr",  # The coordinate order that follows the extracted points. Use yxsr or xysr.
    random_seed=12345,  # The random seed value for TensorFlow and Numpy.
    pyramid_levels=5,  # The number of downsample levels in the pyramid.
    upsampled_levels=1,  # The number of upsample levels in the pyramid.
    scale_factor_levels=np.sqrt(2),  # The scale factor between the pyramid levels.
    scale_factor=2.0,  # The scale factor to extract patches before descriptor.
    # GPU Settings
    gpu_memory_fraction=0.6,  # The fraction of GPU used by the script.
    gpu_visible_devices="0",  # Set CUDA_VISIBLE_DEVICES variable.
):

    print("configuration:", conf)

    # remove verbose bits from tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible_devices

    version_network_name = network_version + conf["name"]

    if not extract_MS:
        pyramid_levels = 0
        upsampled_levels = 0

    print("Extract features for : " + version_network_name)

    aux.check_directory(results_dir)
    aux.check_directory(os.path.join(results_dir, version_network_name))

    def extract_features(image):

        pyramid = pyramid_gaussian(
            image, max_layer=pyramid_levels, downscale=scale_factor_levels
        )

        score_maps = {}
        for (j, resized) in enumerate(pyramid):
            im = resized.reshape(1, resized.shape[0], resized.shape[1], 1)

            feed_dict = {
                input_network: im,
                phase_train: False,
                dimension_image: np.array(
                    [1, im.shape[1], im.shape[2]], dtype=np.int32
                ),
            }

            im_scores = sess.run(maps, feed_dict=feed_dict)

            im_scores = geo_tools.remove_borders(im_scores, borders=border_size)
            score_maps["map_" + str(j + 1 + upsampled_levels)] = im_scores[0, :, :, 0]

        if upsampled_levels:
            for j in range(upsampled_levels):
                factor = scale_factor_levels ** (upsampled_levels - j)
                up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

                im = np.reshape(up_image, (1, up_image.shape[0], up_image.shape[1], 1))

                feed_dict = {
                    input_network: im,
                    phase_train: False,
                    dimension_image: np.array(
                        [1, im.shape[1], im.shape[2]], dtype=np.int32
                    ),
                }

                im_scores = sess.run(maps, feed_dict=feed_dict)

                im_scores = geo_tools.remove_borders(im_scores, borders=border_size)
                score_maps["map_" + str(j + 1)] = im_scores[0, :, :, 0]

        im_pts = []
        for idx_level in range(levels):

            scale_value = scale_factor_levels ** (idx_level - upsampled_levels)
            scale_factor = 1.0 / scale_value

            h_scale = np.asarray(
                [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]]
            )
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = point_level[idx_level]
            if idx_level > 0:
                res_points = int(
                    np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum()
                    - len(im_pts)
                )
                num_points_level = res_points

            im_scores = rep_tools.apply_nms(
                score_maps["map_" + str(idx_level + 1)], nms_size
            )
            im_pts_tmp = geo_tools.get_point_coordinates(
                im_scores, num_points=num_points_level, order_coord="xysr"
            )

            im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

        if order_coord == "yxsr":
            im_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], im_pts)))

        im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
        im_pts = im_pts[:num_points]

        # Extract descriptor from features
        descriptors = []
        im = image.reshape(1, image.shape[0], image.shape[1], 1)
        for idx_desc_batch in range(int(len(im_pts) / 250 + 1)):
            points_batch = im_pts[idx_desc_batch * 250 : (idx_desc_batch + 1) * 250]

            if not len(points_batch):
                break

            feed_dict = {
                input_network: im,
                phase_train: False,
                kpts_coord: points_batch[:, :2],
                kpts_scale: scale_factor * points_batch[:, 2],
                kpts_batch: np.zeros(len(points_batch)),
                dimension_image: np.array(
                    [1, im.shape[1], im.shape[2]], dtype=np.int32
                ),
            }

            patch_batch = sess.run(input_patches, feed_dict=feed_dict)
            patch_batch = np.reshape(patch_batch, (patch_batch.shape[0], 1, 32, 32))
            data_a = torch.from_numpy(patch_batch)
            data_a = data_a.cuda()
            data_a = Variable(data_a)
            with torch.no_grad():
                out_a = model(data_a)
            desc_batch = out_a.data.cpu().numpy().reshape(-1, 128)
            if idx_desc_batch == 0:
                descriptors = desc_batch
            else:
                descriptors = np.concatenate([descriptors, desc_batch], axis=0)

        return im_pts, descriptors

    with tf.Graph().as_default():

        tf.set_random_seed(random_seed)

        with tf.name_scope("inputs"):

            # Define the input tensor shape
            tensor_input_shape = (None, None, None, 1)

            input_network = tf.placeholder(
                dtype=tf.float32, shape=tensor_input_shape, name="input_network"
            )
            dimension_image = tf.placeholder(
                dtype=tf.int32, shape=(3,), name="dimension_image"
            )
            kpts_coord = tf.placeholder(
                dtype=tf.float32, shape=(None, 2), name="kpts_coord"
            )
            kpts_batch = tf.placeholder(
                dtype=tf.int32, shape=(None,), name="kpts_batch"
            )
            kpts_scale = tf.placeholder(dtype=tf.float32, name="kpts_scale")
            phase_train = tf.placeholder(tf.bool, name="phase_train")

        with tf.name_scope("model_deep_detector"):

            deep_architecture = keynet(
                argparse.Namespace(
                    num_levels_within_net=num_levels_within_net,
                    factor_scaling_pyramid=factor_scaling_pyramid,
                    num_learnable_blocks=num_learnable_blocks,
                    num_filters=num_filters,
                    conv_kernel_size=conv_kernel_size,
                    nms_size=nms_size,
                    random_seed=random_seed,
                )
            )
            output_network = deep_architecture.model(
                input_network, phase_train, dimension_image, reuse=False
            )
            maps = tf.nn.relu(output_network["output"])

        # Extract Patches from inputs:
        input_patches = loss_desc.build_patch_extraction(
            kpts_coord, kpts_batch, input_network, kpts_scale=kpts_scale
        )

        # Define Pytorch HardNet
        model = None

        if conf["name"] == "SOSNet":
            model = sosnet_model.SOSNet32x32()
            model.load_state_dict(torch.load(conf["pytorch_weight"]))
        else:
            model = HardNet()
            checkpoint = torch.load(conf["pytorch_weight"])
            model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        model.cuda()

        # Define variables
        detect_var = [v for v in tf.trainable_variables(scope="model_deep_detector")]

        if os.listdir(checkpoint_det_dir):
            (
                init_assign_op_det,
                init_feed_dict_det,
            ) = tf.contrib.framework.assign_from_checkpoint(
                tf.train.latest_checkpoint(checkpoint_det_dir), detect_var
            )

        point_level = []
        tmp = 0.0
        factor_points = scale_factor_levels ** 2
        levels = pyramid_levels + upsampled_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - upsampled_levels))
            point_level.append(
                num_points * factor_points ** (-1 * (idx_level - upsampled_levels))
            )

        point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))

        # GPU Usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            if os.listdir(checkpoint_det_dir):
                sess.run(init_assign_op_det, init_feed_dict_det)

            # read image and extract keypoints and descriptors
            f = open(list_images, "r")
            for path_to_image in f:
                path = path_to_image.split("\n")[0]

                if not os.path.exists(path):
                    print("[ERROR]: File {0} not found!".format(path))
                    return

                create_result_dir(os.path.join(results_dir, version_network_name, path))

                im = read_bw_image(path)

                im = im.astype(float) / im.max()

                im_pts, descriptors = extract_features(im)

                file_name = (
                    os.path.join(results_dir, version_network_name, path) + ".kpt"
                )
                np.save(file_name, im_pts)

                file_name = (
                    os.path.join(results_dir, version_network_name, path) + ".dsc"
                )
                np.save(file_name, descriptors)


# if __name__ == "__main__":
#     extract_multiscale_features()

#!/usr/bin/env python2

import sys
import argparse
import cv2
import os
import numpy as np
import openface


def get_image_representation(image_relative_path, dimension, network_model, align_dlib):
    image = cv2.imread(image_relative_path)
    if image is None:
        raise ValueError("Unable to load image at {}".format(image_relative_path))

    colored_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bounding_box = align_dlib.getLargestFaceBoundingBox(colored_image)
    if not bounding_box:
        raise ValueError("Unable to find a face in provided image {}".format(image_relative_path))

    aligned_face = align_dlib.align(
        dimension,
        colored_image,
        bounding_box,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE
    )

    if aligned_face is None:
        raise ValueError("Unable to align faces in image {} with dimension {}".format(image_relative_path, dimension))

    return network_model.forward(aligned_face)


def generate_profiles(cmd_arguments, folder, network_model, align_dlib):
    number_of_images = len(cmd_arguments.imgs)
    if number_of_images != 3:
        raise ValueError("At least 3 images needed to generate a valid profile")

    print('Generating profile given {} images'.format(number_of_images))

    profile_id = 0

    while os.path.exists(os.path.join(folder, 'P_{}'.format(profile_id))):
        profile_id += 1

    profile_folder = os.path.join(folder, 'P_{}'.format(profile_id))
    os.mkdir(profile_folder)

    for i, img in enumerate(cmd_arguments.imgs):
        try:
            rep = get_image_representation(img, cmd_arguments.imgDim, network_model, align_dlib)
            np.savetxt(os.path.join(os.path.join(profile_folder), '{:03d}'.format(i)), rep)
            print('Generated profile no. {:03d} for {}'.format(i, folder))
        except ValueError as e:
            print('Skipping profile no. {:03d} for {}! {}'.format(i, folder, e.message))


def parse_command_line(root_folder):

    models_folder = os.path.join(root_folder, 'models')
    dlib_model_folder = os.path.join(models_folder, 'dlib')
    openface_folder = os.path.join(models_folder, 'openface')

    parser = argparse.ArgumentParser()
    parser.add_argument('imgs',
                        type=str, nargs='+', help="Input images")
    parser.add_argument('--dlibFacePredictor',
                        type=str, help="Path to dlib face predictor",
                        default=os.path.join(dlib_model_folder, 'shape_predictor_68_face_landmarks.dat'))

    parser.add_argument('--networkModel',
                        type=str, help="Path to Torch Network Model.",
                        default=os.path.join(openface_folder, 'nn4.small2.v1.t7'))

    parser.add_argument('--imgDim',
                        type=int, help="Default image dimension", default=96)

    parser.add_argument('--profile',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.relpath(__file__))

    profiles_folder = os.path.join(current_directory, 'profile')
    if not os.path.exists(profiles_folder):
        os.mkdir(profiles_folder)

    args = parse_command_line(current_directory)

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

    if args.profile:
        generate_profiles(args, profiles_folder, net, align)
    else:
        if len(args.imgs) != 1:
            raise ValueError("At least 1 photo needed for comparison!")

        distances_from_known_profiles = np.zeros(len(os.listdir(profiles_folder)) * 3, dtype=np.float)
        class_profiles = np.empty(len(os.listdir(profiles_folder)) * 3, dtype=np.int8)

        profiles = {}
        counter = 0
        id_class = 0

        representation = get_image_representation(args.imgs[0], args.imgDim, net, align)

        for profile_photo in os.listdir(profiles_folder):
            current_folder = os.path.join(profiles_folder, profile_photo)
            profiles[id_class] = current_folder
            for profile_file in os.listdir(current_folder):
                if profile_file == 'info':
                    continue

                rapBack = np.loadtxt(os.path.join(current_folder, profile_file))

                d = representation - rapBack
                distances_from_known_profiles[counter] = np.dot(d, d)
                class_profiles[counter] = id_class
                counter += 1
            id_class += 1

            best_profiles = np.argsort(distances_from_known_profiles)[:3]
            prof_min = class_profiles[best_profiles]

            if prof_min[0] > 0.5:
                print('Unknown profile!')
                sys.exit(1)

            if prof_min[0] == prof_min[1] or prof_min[0] == prof_min[2]:
                print('Found profile {}'.format(profiles[prof_min[0]]))
                sys.exit(0)

            if prof_min[1] == prof_min[2]:
                print('Found profile {}'.format(profiles[prof_min[1]]))
                sys.exit(0)

            print('Unknown face')
            sys.exit(2)

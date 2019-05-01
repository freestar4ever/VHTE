#!/usr/bin/env python2

import argparse
import cv2
import os
import numpy as np
import openface


def get_aligned_face(image_relative_path, dimension, network_model, align_dlib):
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
    if number_of_images < 3:
        raise ValueError("At least 3 images needed to generate a valid profile")

    profile_name = cmd_arguments.name

    print('Generating profile for {}, given {} images'.format(profile_name, number_of_images))
    os.path.exists(os.path.join(folder, profile_name))

    profile_folder = os.path.join(folder, profile_name)
    os.mkdir(profile_folder)

    for i, img in enumerate(cmd_arguments.imgs):
        try:
            face = get_aligned_face(img, cmd_arguments.imgDim, network_model, align_dlib)
            np.savetxt(os.path.join(os.path.join(profile_folder), '{:03d}'.format(i)), face)
            print('Generated profile no. {:03d} for {}'.format(i, profile_name))
        except ValueError as e:
            print('Skipping profile no. {:03d} for {}! {}'.format(i, profile_name, e.message))


def identify_profile(args, folder, net, align):
    if len(args.imgs) != 1:
        raise ValueError("At least 1 photo needed for comparison!")
    try:
        face_to_found = get_aligned_face(args.imgs[0], args.imgDim, net, align)
    except ValueError:
        print('Unknown input face, cannot get image representation! Exiting')
        return None

    profiles = {}
    id_class = 0

    # print('Finding who is {}'.format(args.imgDim))
    for folder_profile_photos in os.listdir(folder):
        current_folder = os.path.join(folder, folder_profile_photos)
        distances_from_known_profiles = []

        # print('....Finding best match into {}'.format(folder_profile_photos))
        for profile_file in sorted(os.listdir(current_folder)):
            if profile_file == 'info':
                continue

            current_face = np.loadtxt(os.path.join(current_folder, profile_file))
            difference_matrix = face_to_found - current_face

            distances_from_known_profiles.append(np.dot(difference_matrix, difference_matrix))
        id_class += 1

        avg_distance = sum(distances_from_known_profiles) / float(len(os.listdir(folder)))
        profiles[id_class] = (current_folder, avg_distance)

    # print('..Result of scanning:')
    # for p, distance in profiles.items():
    #     print('....{} --> {}'.format(p, distance))

    # order by second value of tuple
    best_match = sorted(profiles.items(), key=lambda kv: kv[1][1])[0]

    # if best_match[1][1] > 0.5:
    #     print('Face not found! Best match is {} with distance {}'.format(best_match[1][0], best_match[1][1]))
    #     return None

    return best_match[1]


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
                        type=int, help="Defclault image dimension", default=96)

    parser.add_argument('--name',
                        type=str, help="Name of profile")

    parser.add_argument('--profile',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.relpath(__file__))

    profiles_folder = os.path.join(current_directory, 'profile')
    if not os.path.exists(profiles_folder):
        os.mkdir(profiles_folder)

    cmd_args = parse_command_line(current_directory)

    aligner = openface.AlignDlib(cmd_args.dlibFacePredictor)
    neural_net = openface.TorchNeuralNet(cmd_args.networkModel, cmd_args.imgDim)

    if cmd_args.profile:
        generate_profiles(cmd_args, profiles_folder, neural_net, aligner)
    else:
        who = identify_profile(cmd_args, profiles_folder, neural_net, aligner)
        if who is not None:
            print('Found! is: {} (with avg distance of {})'.format(who[0], who[1]))
        else:
            print('Cannot detect who is {}'.format(cmd_args.imgs))

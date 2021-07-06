"""
The retrieve module downloads machine learning datasets from popular
repositories.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
import custom  # Functions for retrieve arbitrary datasets
import itertools  # Functions creating iterations for efficient looping
import torchvision  # Datasets, transforms and Models specific to Computer Vision
import tensorflow_datasets  # A collection of ready-to-use datasets
from utils import print  # use timestamped print

# TODO
# - resolve datasets that require labels to be downloaded separetely
# - add print statements


class Downloader:
    """
    This downloader class serves as a wrapper for popular
    machine learning libraries to retrieve datasets. Moreover,
    it is designed to be easily extendable to support downloading
    datasets from ostensibly any location via the Requests module.

    :func:`__init__`: instantiates Downloader objects
    :func:`custom`: defines an interface for custom dataset downloaders
    :func:`pytorch`: retrieve datasets from torchvision
    :func:`tensorflow`: retreive datasets from tensorflow
    :func:`uci`: retreive datasets from UCI machine learning repository
    """

    def __init__(self, datasets):
        """
        This function initializes the supported datasets from PyTorch,
        TensorFlow, and user-specified datasets as described in the custom.py
        module. Details pertaining to the libraries are described below.

        -- Custom --
        The Custom class defines an interface for which the Downloader class
        can consume and use to retrieve the desired dataset. As the interfaces
        to retrieve these datasets can be entirely abitrary, it is on the user
        to write templates that can properly process the dataset. As the
        custom.py module describes, the only components that need to be
        well-defined are (1) the URL to retrieve the dataset, (2) any
        preprocessing directives, and (3) methods to read the dataset such that
        it can be prepared into a numpy array. The supported datasets are described
        in custom.py.

        -- PyTorch --
        The datasets in PyTorch have non-standardized interfaces. Thus,
        supported datasets are encoded as dictionaries of dictionaries, with
        the dataset names as keys and, as values, a dictionary containing two
        keys: "name", which maps the dataset name to the case-sensitive module
        name; and, "split", which maps the the dataset "category" (e.g.,
        training, testing, validation, landmarks, outlines, etc.) parameter and
        possible values as a tuple.

        At this time, the following PyTorch datasets are not supported:
        - Cityscapes (does not support downloading)
        - MS Coco Captions (does not support downloading)
        - MS Coco (does not support downloading)
        - EMNIST (incompatible with this library)
        - Flickr8k (does not support downloading)
        - Flickr30k (does not support downloading)
        - HMDB51 (does not support downloading)
        - Kinetics-400 (incompatible with this library)
        - UCF101 (does not support downloading)

        -- TensorFlow --
        For TensorFlow, the interfaces are largely standardized and
        well-defined. To this end, little additional effort is needed to
        integrate the supported datasets into this framework. Thus, the only
        datasets that are excluded are those that do not support downloading.
        Moreover, this repo assumes the stable version of tensorflow-datasets.

        At this time, the following TensorFlow datasets are not supported:
        Audio
        - dementiabank (does not support downloading)
        - savee (does not support downloading)
        - voxceleb (does not support downloading)
        - voxforge (does not support downloading)

        Images
        - abstract_reasoning (does not support downloading)
        - celeb_a_hq (does not support downloading)
        - cityscapes (does not support downloading)

        Image Classification
        - curated_breast_imaging_ddsm (does not support downloading)
        - diabetic_retinopathy_detection (does not support downloading)
        - dmlab (does not support downloading)
        - imagenet2012 (does not support downloading)
        - imagenet2012_corrupted (does not support downloading)
        - imagenet2012_real (does not support downloading)
        - imagenet2012_subset (does not support downloading)
        - resisc45 (does not support downloading)
        - waymo_open_dataset (requires authorization and registration)

        Summarization
        - covid19sum (does not support downloading)
        - newsroom (does not support downloading)
        - samsum (does not support downloading)
        - wikihow (does not support downloading)
        - xsum (does not support downloading)

        Text
        - c4 (does not support downloading)
        - reddit_disentanglement (does not support downloading)
        - story_cloze (does not support downloading)

        Translation
        - wmt13_translate (does not support downloading)
        - wmt14_translate (does not support downloading)
        - wmt15_translate (does not support downloading)
        - wmt16_translate (does not support downloading)
        - wmt17_translate (does not support downloading)
        - wmt18_translate (does not support downloading)
        - wmt19_translate (does not support downloading)
        - wmt_t2t_translate (does not support downloading)

        Video
        - tao (does not support downloading)
        - youtube_vis (does not support downloading)

        Vision Language
        - gref (does not support downloading)

        :param datasets: datasets to download
        :type datasets: list of strings
        :return: downloader
        :rtype: Downloader object
        """
        self.datasets = datasets

        # define supported custom datasets

        # define supported pytorch datasets
        self.pytorch_datasets = {
            "caltech101": {
                "name": "Caltech101",
                "split": ("target_type", ["category", "annotation"]),
            },
            "caltech256": {
                "name": "Caltech256",
                "split": None,
            },
            "celeba": {
                "name": "CelebA",
                "split": ("split", ["all"]),
            },
            "cifar10": {
                "name": "CIFAR10",
                "split": ("train", [True, False]),
            },
            "cifar100": {
                "name": "CIFAR100",
                "split": ("train", [True, False]),
            },
            "fakedata": {
                "name": "FakeData",
                "split": None,
            },
            "fashionmnist": {
                "name": "FashionMNIST",
                "split": ("train", [True, False]),
            },
            "imagenet": {
                "name": "ImageNet",
                "split": ("split", ["train", "val"]),
            },
            "kitti": {
                "name": "Kitti",
                "split": ("train", [True, False]),
            },
            "kmnist": {
                "name": "KMNIST",
                "split": ("train", [True, False]),
            },
            "lsun": {
                "name": "LSUN",
                "split": ("classes", ["train", "val", "test"]),
            },
            "mnist": {
                "name": "MNIST",
                "split": ("train", [True, False]),
            },
            "omniglot": {
                "name": "Omniglot",
                "split": ("background", [True, False]),
            },
            "phototour": {
                "name": "PhotoTour",
                "split": ("name", ["notredame", "yosemite", "liberty"]),
            },
            "places365": {
                "name": "Places365",
                "split": ("split", ["train-standard", "train-challenge", "val"]),
            },
            "qmnist": {
                "name": "QMNIST",
                "split": ("train", [True, False]),
            },
            "sbdtaset": {
                "name": "SBDataset",
                "split": ("image_set", ["train", "val"]),
            },
            "sbu": {
                "name": "SBU",
                "split": None,
            },
            "semeion": {
                "name": "SEMEION",
                "split": None,
            },
            "stl10": {
                "name": "STL10",
                "split": ("split", ["train", "test"]),
            },
            "svhn": {
                "name": "SVHN",
                "split": ("split", ["train", "test"]),
            },
            "usps": {
                "name": "USPS",
                "split": ("train", [True, False]),
            },
            "vocsegmentation": {
                "name": "VOCSegmentation",
                "split": ("image_set", ["train", "val"]),
            },
            "vocdetection": {
                "name": "VOCDetection",
                "split": ("image_set", ["train", "val"]),
            },
            "widerface": {
                "name": "WIDERFace",
                "split": ("split", ["train", "val", "test"]),
            },
        }

        # define supported tensorflow datasets
        self.tensorflow_datasets = {
            # audio
            "accentdb",
            "common_voice",
            "crema_d",
            "fuss",
            "groove",
            "gtzan",
            "gtzan_music_speech",
            "librispeech",
            "libritts",
            "ljspeech",
            "nsynth",
            "speech_commands",
            "spoken_digit",
            "tedlium",
            "vctk",
            "yes_no",
            # graphs
            "ogbg_molpcba",
            # images
            "aflw2k3d",
            "arc",
            "bccd",
            "binarized_mnist",
            "celeb_a",
            "clevr",
            "clic",
            "coil100",
            "div2k",
            "downsampled_imagenet",
            "dsprites",
            "duke_ultrasound",
            "flc",
            "lost_and_found",
            "lsun",
            "nyu_depth_v2",
            "s3o4d",
            "scene_parse150",
            "shapes3d",
            "the300w_lp",
            # image classification
            "beans",
            "bigearthnet",
            "binary_alpha_digits",
            "caltech101",
            "caltech_birds2010",
            "caltech_birds2011",
            "cars196",
            "cassava",
            "cats_vs_dogs",
            "cifar10",
            "cifar100",
            "cifar10_1",
            "cifar10_corrupted",
            "citrus_leaves",
            "cmaterdb",
            "colorectal_histology",
            "colorectal_histology_large",
            "cycle_gan",
            "deep_weeds",
            "dtd",
            "emnist",
            "eurostat",
            "fashion_mnist",
            "food101",
            "geirhos_conflict_stimuli",
            "horses_or_humans",
            "i_naturalist2017",
            "imagenet_a",
            "imagenet_r",
            "imagenet_resized",
            "imagenet_v2",
            "imagenette",
            "imagewang",
            "kmnist",
            "lfw",
            "malaria",
            "mnist",
            "mnist_corrupted",
            "omniglot",
            "oxford_flowers102" "oxford_iiit_pet" "patch_camelyon",
            "pet_finder",
            "places365_small",
            "plant_leaves",
            "plant_village",
            "plantae_k",
            "quickdraw_bitmap",
            "rock_paper_scissors",
            "siscore",
            "smallnorb",
            "so2sat",
            "stanford_dogs",
            "standford_online_products",
            "stl10",
            "sun397",
            "svhn_cropped",
            "tf_flowers",
            "uc_merced",
            "visual_domain_decathlon",
            # object detection
            "coco",
            "coco_captions",
            "kitti",
            "lvis",
            "open_images_challenge2019_detection",
            "open_images_v4",
            "voc",
            "wider_face",
            # question answering
            "ai2_arc",
            "ai2_arc_with_ir",
            "coqa",
            "cosmos_qa",
            "mctaco",
            "mlqa",
            "natural_question",
            "natural_questions_open" "qasc",
            "squad",
            "trivia_qa",
            "tydi_qa",
            "web_questions",
            "xquad",
            # structured
            "cherry_blossoms",
            "dart",
            "e2e_cleaned",
            "efron_morris75",
            "forest_fires",
            "genomics_ood",
            "german_credit_numeric",
            "higgs",
            "howell",
            "iris",
            "movie_lens",
            "movielens",
            "radon",
            "rock_you",
            "titanic",
            "web_nlg",
            "wiki_bio",
            "wiki_table_questions",
            "wiki_table_text",
            "wine_quality",
            # summarization
            "aeslc",
            "big_patent",
            "billsum",
            "cnn_dailymail",
            "gigaword",
            "multi_news",
            "opinion_abstracts",
            "opinosis",
            "reddit",
            "reddit_tifu",
            "scientific_papers",
            # text
            "ag_news_subsets",
            "anli",
            "blimp",
            "bool_q",
            "cfq",
            "civil_comments",
            "clinc_oos",
            "cos_e",
            "definite_pronoun_resolution",
            "dolphin_umber_word",
            "drop",
            "eraser_multi_rc",
            "esnli",
            "gap",
            "gem",
            "glue",
            "goemotions",
            "gpt3",
            "hellaswag",
            "imdb_reviews",
            "irc_disentanglement",
            "lambada",
            "librispeech_lm",
            "lm1b",
            "math_dataset",
            "movie_rationales",
            "multi_nli",
            "multi_nli_mismatch",
            "openbookqa",
            "paws_wiki",
            "paws_x_wiki",
            "pg19",
            "piqa",
            "qa4mre",
            "quac",
            "race",
            "salient_span_wikipedia",
            "scan",
            "schema_guided_dialogue",
            "scicite",
            "sentiment140",
            "snli",
            "star_cfq",
            "super_glue",
            "tiny_shapespeare",
            "trec",
            "wiki40b",
            "wikiann",
            "wikipedia",
            "wikipedia_toxicity_subtypes",
            "winogrande",
            "wordnet",
            "wsc273",
            "xnli",
            "xtreme_pawsx",
            "xtreme_xnli",
            "yelp_polarity_reviews",
            # translation
            "flores",
            "opus",
            "para_crawl",
            "ted_hrlr_translate",
            "ted_multi_translate",
            # video
            "bair_robot_pushing_small",
            "davis",
            "moving_mnist",
            "robonet",
            "starcraft_video",
            "ucf101",
        }

        # define pytorch dataset category mappings
        self.pytorch_map = {True: "train", False: "test"}
        return None

    def custom(self, dataset):
        """"""
        return

    def download(self, datasets):
        """
        This function dispatches dataset downloads to the respective handlers.

        :param datasets: datasets to download
        :type datasets: list of strings
        :return: the downloaded datasets
        :rtype: dictionary of datasets containing dictionaries of categories
        """
        downloads = {}
        for dataset in datasets:
            if dataset in self.pytorch_datasets:
                downloads[dataset] = self.pytorch(
                    self.pytorch_datasets[dataset]["name"],
                    *self.pytorch_datasets[dataset]["split"]
                )
            elif dataset in self.tensorflow_datasets:
                downloads[dataset] = self.tensorflow(dataset)
            elif dataset in self.custom_datasets:
                pass
            else:
                raise KeyError(dataset, "not supported")
        return downloads

    def pytorch(self, dataset, arg, splits, directory="/tmp/"):
        """
        This function serves as a wrapper for torchvision.datasets
        (https://pytorch.org/vision/stable/datasets.html). While this API is
        designed to be as standardized as possible, many of the datasets
        implement their own custom API (since the parameters and the values
        they can take are defined by the dataset authors). Specifically, this
        function: (1) downloads the entire dataset, (2) saves it in /tmp/, (3)
        returns the dataset as a numpy array.

        :param dataset: a dataset from torchvision.datasets
        :type dataset: string
        :param arg: the name of the argument governing the dataset splits
        :type arg: string
        :param splits: list of dataset "categories" to download
        :type splits: list or NoneType
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy versions of the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """
        return {
            # map splits to strings so that they are human-readable
            self.pytorch_map.get(split, split): getattr(torchvision.datasets, dataset)(
                # use keyword arguments since interfaces can differ slightly
                **{
                    "root": directory,
                    "download": True,
                    "transform": torchvision.transforms.ToTensor(),
                    arg: split,
                }
            )
            for split in splits
        }

    def tensorflow(self, dataset, directory="/tmp/"):
        """
        This function serves as a wrapper for tensorflow_datasets.
        (https://www.tensorflow.org/datasets). The interfaces for the vast
        majority of datasets are identical, and thus, this wrapper largely
        prepares the data such that it conforms to the standard used throughout
        the rest of this repository.

        :param dataset: a dataset from tensorflow_datasets
        :type dataset: string
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy versions of the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """
        return tensorflow_datasets.as_numpy(
            tensorflow_datasets.load(dataset, data_dir=directory, batch_size=-1)
        )


if __name__ == "__main__":
    """
    Example usage of MachineLearningDataSets via the command-line as:

        $ mlds mnist nslkdd -i 50-700 -i protocol flag --outdir datasets
            -n mnist_mod nslkdd_mod -s normalization -s standardization minmax
            -a --destupefy

    This (1) downloads MNIST and NSL-KDD, (2) selects all features for MNIST
    (necessary to correctly associate the following "-i" with NSL-KDD) and
    "protocol" & "flag" for NSL-KDD, (3) specifies an alternative output
    directory (instead of "out/"), (4) changes the base dataset name when
    saved, (5) applies minmax scaling to MNIST and creates two copies of the
    NSL-KDD that are standardization & normalized, respectively, and (6)
    computes basic analytics and applies destupification (to both datasets).
    """
    raise SystemExit(0)

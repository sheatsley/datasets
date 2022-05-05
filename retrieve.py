"""
The retrieve module downloads machine learning datasets from popular
repositories.
Author: Ryan Sheatsley
Fri Jun 18 2021
"""
import adapters  # Third-party datasets
import pandas  # Python Data Analysis Library
import torchvision  # Datasets, transforms and Models specific to Computer Vision
import tensorflow_datasets  # A collection of ready-to-use datasets

# TODO
# add unit tests
# augment tensorflow parser to include structured data (generalize image key)
# convert dataset definitions from strings to callables


class Downloader:
    """
    This Downloader class serves as a wrapper for popular machine learning
    libraries to retrieve datasets. Moreover, it is designed to be easily
    extendable to support downloading datasets from ostensibly any location via
    subclassing BaseAdapter.

    :func:`__init__`: instantiates Downloader objects
    :func:`adapter`: defines an interface for third-party dataset downloaders
    :func:`prep`: prepares pytorch and tensorflow into usable pandas dataframes
    :func:`pytorch`: retrieve datasets from torchvision
    :func:`tensorflow`: retreive datasets from tensorflow
    """

    def __init__(self, dataset):
        """
        This method initializes the supported datasets from PyTorch,
        TensorFlow, and user-specified datasets as described in the custom.py
        module. Details pertaining to the libraries are described below.

        -- Adapters --
        The BaseAdapter class defines an interface this class can consume and
        use to retrieve the desired dataset. As the interfaces to retrieve
        these datasets can be entirely abitrary, it is on the user to write
        templates that can properly process the dataset. As the BaseAdapter
        class describes, the only components that need to be well-defined are
        (1) the URL to retrieve the dataset, (2) methods to read the dataset,
        and (3) any preprocessing directives such that it can be prepared into
        a pandas dataframe. The supported datasets are:

        Malware Detection
        - CIC-MalMem-2022

        Network Intrusion Detection
        - NSL-KDD
        - UNSW-NB15

        Phishing Detection
        - Phishing dataset

        -- PyTorch --
        The datasets in PyTorch have non-standardized interfaces. Thus,
        supported datasets are encoded as dictionaries of dictionaries, with
        the dataset names as keys and, as values, a dictionary containing two
        keys: "name", which maps the dataset name to the case-sensitive module
        name; and, "part", which maps the the dataset "category" (e.g.,
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

        :param datasets: dataset to download
        :type datasets: string
        :return: dataset retriever
        :rtype: Downloader object
        """
        self.dataset = dataset

        # define adapter datasets
        self.adapter_datasets = {
            dset.__name__.lower(): dset for dset in adapters.available
        }

        # define supported pytorch datasets
        self.pytorch_datasets = {
            "caltech101": {
                "name": "Caltech101",
                "part": ("target_type", ["category", "annotation"]),
            },
            "caltech256": {
                "name": "Caltech256",
                "part": None,
            },
            "celeba": {
                "name": "CelebA",
                "part": ("split", ["all"]),
            },
            "cifar10": {
                "name": "CIFAR10",
                "part": ("train", [True, False]),
            },
            "cifar100": {
                "name": "CIFAR100",
                "part": ("train", [True, False]),
            },
            "fakedata": {
                "name": "FakeData",
                "part": None,
            },
            "fashionmnist": {
                "name": "FashionMNIST",
                "part": ("train", [True, False]),
            },
            "imagenet": {
                "name": "ImageNet",
                "part": ("split", ["train", "val"]),
            },
            "kitti": {
                "name": "Kitti",
                "part": ("train", [True, False]),
            },
            "kmnist": {
                "name": "KMNIST",
                "part": ("train", [True, False]),
            },
            "lsun": {
                "name": "LSUN",
                "part": ("classes", ["train", "val", "test"]),
            },
            "mnist": {
                "name": "MNIST",
                "part": ("train", [True, False]),
            },
            "omniglot": {
                "name": "Omniglot",
                "part": ("background", [True, False]),
            },
            "phototour": {
                "name": "PhotoTour",
                "part": ("name", ["notredame", "yosemite", "liberty"]),
            },
            "places365": {
                "name": "Places365",
                "part": ("split", ["train-standard", "train-challenge", "val"]),
            },
            "qmnist": {
                "name": "QMNIST",
                "part": ("train", [True, False]),
            },
            "sbdtaset": {
                "name": "SBDataset",
                "part": ("image_set", ["train", "val"]),
            },
            "sbu": {
                "name": "SBU",
                "part": None,
            },
            "semeion": {
                "name": "SEMEION",
                "part": None,
            },
            "stl10": {
                "name": "STL10",
                "part": ("split", ["train", "test"]),
            },
            "svhn": {
                "name": "SVHN",
                "part": ("split", ["train", "test"]),
            },
            "usps": {
                "name": "USPS",
                "part": ("train", [True, False]),
            },
            "vocsegmentation": {
                "name": "VOCSegmentation",
                "part": ("image_set", ["train", "val"]),
            },
            "vocdetection": {
                "name": "VOCDetection",
                "part": ("image_set", ["train", "val"]),
            },
            "widerface": {
                "name": "WIDERFace",
                "part": ("split", ["train", "val", "test"]),
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
        return None

    def adapter(self, dataset, directory="/tmp/"):
        """
        This method instantiates objects from the adapters package to retrieve
        datasets from arbitrary network resources. It relies on the request
        module for retrieving the dataset. Specific details of the templates
        can be found in BaseAdapter.

        :param dataset: dataset defined in adapters package
        :type dataset: BaseAdapter subclass
        :param directory: directory to download the datasets to
        :type directory: string
        :return: pandas dataframes representing the dataset
        :rtype: dictionary; keys are dataset types & values are dataframes
        """
        return dataset(directory=directory).read()

    def download(self):
        """
        This method dispatches dataset downloads to the respective handlers.

        :param dataset: dataset to download
        :type dataset: string
        :return: the downloaded dataset
        :rtype: dictionary; keys are dataset types & values are dataframes
        """
        if self.dataset in self.pytorch_datasets:
            return self.prep(
                self.pytorch(
                    self.pytorch_datasets[self.dataset]["name"],
                    *self.pytorch_datasets[self.dataset]["part"],
                )
            )
        elif self.dataset in self.tensorflow_datasets:
            return self.prep(self.tensorflow(self.dataset))
        elif self.dataset in self.adapter_datasets:
            return self.adapter(self.adapter_datasets[self.dataset])
        else:
            raise KeyError(self.dataset, "not supported")

    def prep(self, dataset):
        """
        This method serves as a helper function for preparing datasets
        retrieved from PyTorch or TensorFlow datasets. Specifically, it: (1)
        flattens the arrays (and stores the original feature shape), (2) casts
        into a pandas dataframe, (3) sets the columns to be strings, and (4)
        saves the original feature shape as "fshape"

        :param dataset: the dataset to prepare
        :type dataset: dictionary; keys as dataset types & values as numpy arrays
        :return: mlds-ready dataset
        :type: dictionary; keys as dataset types & values as pandas dataframes
        """
        fshape = dataset[next(iter(dataset))]["data"].shape[1:]
        for data in dataset.values():
            data["data"] = pandas.DataFrame(data["data"].reshape(len(data["data"]), -1))
            data["data"].columns = data["data"].columns.map(str)
            data["labels"] = pandas.Series(data["labels"], name="class")
        dataset["fshape"] = fshape
        return dataset

    def pytorch(self, dataset, arg, parts, directory="/tmp/"):
        """
        This method serves as a wrapper for torchvision.datasets
        (https://pytorch.org/vision/stable/datasets.html). While this API is
        designed to be as standardized as possible, many of the datasets
        implement their own custom API (since the parameters and the values
        they can take are defined by the dataset authors). Specifically, this
        method: (1) downloads the entire dataset, (2) saves it in /tmp/, (3)
        returns the dataset as a dictionary of numpy arrays.

        :param dataset: a dataset from torchvision.datasets
        :type dataset: string
        :param arg: the name of the argument governing the dataset partitions
        :type arg: string
        :param parts: list of dataset "categories" to download
        :type parts: list or NoneType
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy arrays representing the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """

        # download and cast to numpy arrays (correct True & False partition names)
        torchmap = {True: "train", False: "test"}
        tvds = {
            part: getattr(torchvision.datasets, dataset)(
                # use keyword arguments since interfaces can differ slightly
                **{
                    "root": directory,
                    "download": True,
                    arg: part,
                }
            )
            for part in parts
        }
        return {
            torchmap.get(part, part): {
                "data": data.numpy()
                if hasattr(data := tvds[part].data, "numpy")
                else data,
                "labels": labels.numpy()
                if hasattr(labels := tvds[part].targets, "numpy")
                else labels,
            }
            for part in parts
        }

    def tensorflow(self, dataset, directory="/tmp/"):
        """
        This method serves as a wrapper for tensorflow_datasets.
        (https://www.tensorflow.org/datasets). The interfaces for the vast
        majority of datasets are identical, and thus, this wrapper largely
        prepares the data such that it conforms to the standard used throughout
        the rest of this repository (that is, (1) downloads the entire dataset,
        (2) saves it in /tmp/, and (3) returns the dataset as a dictionary of
        numpy arrays.

        :param dataset: a dataset from tensorflow_datasets
        :type dataset: string
        :param directory: directory to download the datasets to
        :type directory: string
        :return: numpy arrays representing the dataset
        :rtype: dictionary; keys are dataset types & values are numpy arrays
        """

        # download and cast to numpy arrays
        tfds = tensorflow_datasets.load(dataset, data_dir=directory, batch_size=-1)
        return {
            part: {
                "data": tfds[part]["image"].numpy(),
                "labels": tfds[part]["label"].numpy(),
            }
            for part in ("train", "test")
        }


if __name__ == "__main__":
    """ """
    raise SystemExit(0)

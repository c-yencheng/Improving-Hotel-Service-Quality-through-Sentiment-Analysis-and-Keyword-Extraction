{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a23561eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "#  read csv\n",
    "df = pd.read_csv(\"C:/Users/drrr8/OneDrive - Washington University in St. Louis/Desktop/tripadvisor_hotel_reviews.csv\")\n",
    "\n",
    "# convert csv to pkl\n",
    "df.to_pickle('trip.pkl')\n",
    "df = pd.read_pickle('trip.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bc0acbe4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# split the dataset into two\n",
    "\n",
    "# training and test dataset\n",
    "train_data, test_data = train_test_split(df, test_size=0.3, random_state=2)\n",
    "\n",
    "with open('train.pkl', 'wb') as f:\n",
    "    pickle.dump(train_data, f)\n",
    "    \n",
    "with open('test.pkl', 'wb') as f:\n",
    "    pickle.dump(test_data, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e03c9e5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dimensions for test data: (6148, 2)\n",
      "Dimensions for training data: (14343, 2) \n",
      "\n",
      "First 5 rows in test dataset: \n",
      "                                                   Review  Rating\n",
      "10271  not family friendly hotel, travelled wife 2 yr...       1\n",
      "7142   did n't know expect honemoon surprise loved, r...       4\n",
      "18500  enthusiastically reccomend majestic colonial, ...       5\n",
      "17145  problems start beware hotel, choices resort pu...       1\n",
      "2704   enjoyed second stay husband just returned seco...       5 \n",
      "\n",
      "First 5 rows in  training dataset: \n",
      "                                                   Review  Rating\n",
      "5098   nice hotel good service stayed quite hotel par...       4\n",
      "7162   notch resort wife honeymoon secrets 10/6-10/14...       4\n",
      "17228  baihia principe resort friend just returned pu...       4\n",
      "1691   good not good stayed university tower hotel 3 ...       3\n",
      "6805   excellent husband stayed resort october 2003 b...       5\n"
     ]
    }
   ],
   "source": [
    "print(\"Dimensions for test data:\", test_data.shape)\n",
    "print(\"Dimensions for training data:\", train_data.shape,\"\\n\")\n",
    "print(\"First 5 rows in test dataset: \\n\", test_data.head(),\"\\n\")\n",
    "print(\"First 5 rows in  training dataset: \\n\", train_data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a775aa59",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pyLDAvis in d:\\anaconda\\setup\\lib\\site-packages (3.4.1)\n",
      "Collecting sklearn\n",
      "  Downloading sklearn-0.0.post4.tar.gz (3.6 kB)\n",
      "  Preparing metadata (setup.py): started\n",
      "  Preparing metadata (setup.py): finished with status 'done'\n",
      "Requirement already satisfied: pandas>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: gensim in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (4.1.2)\n",
      "Requirement already satisfied: numpy>=1.24.2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.24.3)\n",
      "Requirement already satisfied: scipy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.10.1)\n",
      "Requirement already satisfied: funcy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0)\n",
      "Requirement already satisfied: joblib>=1.2.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.2.0)\n",
      "Requirement already satisfied: numexpr in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.8.3)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (65.5.0)\n",
      "Requirement already satisfied: scikit-learn>=1.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.0.2)\n",
      "Requirement already satisfied: jinja2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.11.3)\n",
      "Requirement already satisfied: tzdata>=2022.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2023.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2022.1)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from scikit-learn>=1.0.0->pyLDAvis) (2.2.0)\n",
      "Requirement already satisfied: smart-open>=1.8.1 in d:\\anaconda\\setup\\lib\\site-packages (from gensim->pyLDAvis) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in d:\\anaconda\\setup\\lib\\site-packages (from jinja2->pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: packaging in d:\\anaconda\\setup\\lib\\site-packages (from numexpr->pyLDAvis) (21.3)\n",
      "Requirement already satisfied: six>=1.5 in d:\\anaconda\\setup\\lib\\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->pyLDAvis) (1.16.0)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in d:\\anaconda\\setup\\lib\\site-packages (from packaging->numexpr->pyLDAvis) (3.0.9)\n",
      "Building wheels for collected packages: sklearn\n",
      "  Building wheel for sklearn (setup.py): started\n",
      "  Building wheel for sklearn (setup.py): finished with status 'done'\n",
      "  Created wheel for sklearn: filename=sklearn-0.0.post4-py3-none-any.whl size=2957 sha256=bf8c32e85e07f7d0f629fa7b3795ec0e505244f067c4a3d18eb92429604adacf\n",
      "  Stored in directory: c:\\users\\drrr8\\appdata\\local\\pip\\cache\\wheels\\d5\\b2\\a9\\590d15767d34955f20a9a033e8db973b79cb5672d95790c0a9\n",
      "Successfully built sklearn\n",
      "Installing collected packages: sklearn\n",
      "Successfully installed sklearn-0.0.post4\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pyLDAvis sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "324a9cc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy in d:\\anaconda\\setup\\lib\\site-packages (1.24.3)\n",
      "Requirement already satisfied: pandas in d:\\anaconda\\setup\\lib\\site-packages (2.0.1)\n",
      "Requirement already satisfied: numpy>=1.20.3 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (1.24.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2022.1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in d:\\anaconda\\setup\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: nltk in d:\\anaconda\\setup\\lib\\site-packages (3.7)\n",
      "Requirement already satisfied: regex>=2021.8.3 in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (2022.7.9)\n",
      "Requirement already satisfied: tqdm in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (4.64.1)\n",
      "Requirement already satisfied: click in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (8.0.4)\n",
      "Requirement already satisfied: joblib in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (1.2.0)\n",
      "Requirement already satisfied: colorama in d:\\anaconda\\setup\\lib\\site-packages (from click->nltk) (0.4.5)\n",
      "Requirement already satisfied: html.parser in d:\\anaconda\\setup\\lib\\site-packages (0.2)\n",
      "Requirement already satisfied: ply in d:\\anaconda\\setup\\lib\\site-packages (from html.parser) (3.11)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package omw-1.4 to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package omw-1.4 is already up-to-date!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pattern3 in d:\\anaconda\\setup\\lib\\site-packages (3.0.0)\n",
      "Requirement already satisfied: pdfminer3k in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (1.3.4)\n",
      "Requirement already satisfied: simplejson in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (3.19.1)\n",
      "Requirement already satisfied: cherrypy in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (18.8.0)\n",
      "Requirement already satisfied: pdfminer.six in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (20221105)\n",
      "Requirement already satisfied: beautifulsoup4 in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (4.11.1)\n",
      "Requirement already satisfied: feedparser in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (6.0.10)\n",
      "Requirement already satisfied: docx in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (0.2.4)\n",
      "Requirement already satisfied: soupsieve>1.2 in d:\\anaconda\\setup\\lib\\site-packages (from beautifulsoup4->pattern3) (2.3.2.post1)\n",
      "Requirement already satisfied: jaraco.collections in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (4.1.0)\n",
      "Requirement already satisfied: portend>=2.1.1 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (3.1.0)\n",
      "Requirement already satisfied: pywin32>=227 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (302)\n",
      "Requirement already satisfied: more-itertools in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (9.1.0)\n",
      "Requirement already satisfied: cheroot>=8.2.1 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (9.0.0)\n",
      "Requirement already satisfied: zc.lockfile in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (3.0.post1)\n",
      "Requirement already satisfied: Pillow>=2.0 in d:\\anaconda\\setup\\lib\\site-packages (from docx->pattern3) (9.2.0)\n",
      "Requirement already satisfied: lxml in d:\\anaconda\\setup\\lib\\site-packages (from docx->pattern3) (4.9.1)\n",
      "Requirement already satisfied: sgmllib3k in d:\\anaconda\\setup\\lib\\site-packages (from feedparser->pattern3) (1.0.0)\n",
      "Requirement already satisfied: cryptography>=36.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer.six->pattern3) (38.0.1)\n",
      "Requirement already satisfied: charset-normalizer>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer.six->pattern3) (2.0.4)\n",
      "Requirement already satisfied: ply in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer3k->pattern3) (3.11)\n",
      "Requirement already satisfied: jaraco.functools in d:\\anaconda\\setup\\lib\\site-packages (from cheroot>=8.2.1->cherrypy->pattern3) (3.6.0)\n",
      "Requirement already satisfied: six>=1.11.0 in d:\\anaconda\\setup\\lib\\site-packages (from cheroot>=8.2.1->cherrypy->pattern3) (1.16.0)\n",
      "Requirement already satisfied: cffi>=1.12 in d:\\anaconda\\setup\\lib\\site-packages (from cryptography>=36.0.0->pdfminer.six->pattern3) (1.15.1)\n",
      "Requirement already satisfied: tempora>=1.8 in d:\\anaconda\\setup\\lib\\site-packages (from portend>=2.1.1->cherrypy->pattern3) (5.2.2)\n",
      "Requirement already satisfied: jaraco.text in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.collections->cherrypy->pattern3) (3.11.1)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda\\setup\\lib\\site-packages (from zc.lockfile->cherrypy->pattern3) (65.5.0)\n",
      "Requirement already satisfied: pycparser in d:\\anaconda\\setup\\lib\\site-packages (from cffi>=1.12->cryptography>=36.0.0->pdfminer.six->pattern3) (2.21)\n",
      "Requirement already satisfied: pytz in d:\\anaconda\\setup\\lib\\site-packages (from tempora>=1.8->portend>=2.1.1->cherrypy->pattern3) (2022.1)\n",
      "Requirement already satisfied: inflect in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (6.0.4)\n",
      "Requirement already satisfied: autocommand in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (2.2.2)\n",
      "Requirement already satisfied: jaraco.context>=4.1 in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (4.3.0)\n",
      "Requirement already satisfied: pydantic>=1.9.1 in d:\\anaconda\\setup\\lib\\site-packages (from inflect->jaraco.text->jaraco.collections->cherrypy->pattern3) (1.10.7)\n",
      "Requirement already satisfied: typing-extensions>=4.2.0 in d:\\anaconda\\setup\\lib\\site-packages (from pydantic>=1.9.1->inflect->jaraco.text->jaraco.collections->cherrypy->pattern3) (4.3.0)\n",
      "Requirement already satisfied: pyLDAvis in d:\\anaconda\\setup\\lib\\site-packages (3.4.1)\n",
      "Requirement already satisfied: numpy>=1.24.2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.24.3)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (65.5.0)\n",
      "Requirement already satisfied: gensim in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (4.1.2)\n",
      "Requirement already satisfied: scikit-learn>=1.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.0.2)\n",
      "Requirement already satisfied: scipy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.10.1)\n",
      "Requirement already satisfied: numexpr in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.8.3)\n",
      "Requirement already satisfied: joblib>=1.2.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.2.0)\n",
      "Requirement already satisfied: jinja2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.11.3)\n",
      "Requirement already satisfied: pandas>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: funcy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0)\n",
      "Requirement already satisfied: pytz>=2020.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2022.1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2023.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2.8.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from scikit-learn>=1.0.0->pyLDAvis) (2.2.0)\n",
      "Requirement already satisfied: smart-open>=1.8.1 in d:\\anaconda\\setup\\lib\\site-packages (from gensim->pyLDAvis) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in d:\\anaconda\\setup\\lib\\site-packages (from jinja2->pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: packaging in d:\\anaconda\\setup\\lib\\site-packages (from numexpr->pyLDAvis) (21.3)\n",
      "Requirement already satisfied: six>=1.5 in d:\\anaconda\\setup\\lib\\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->pyLDAvis) (1.16.0)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in d:\\anaconda\\setup\\lib\\site-packages (from packaging->numexpr->pyLDAvis) (3.0.9)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package averaged_perceptron_tagger is already up-to-\n",
      "[nltk_data]       date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  ['<', 'p', '>', 'The', 'circus', 'dog', 'in', 'a', 'plissé', 'skirt', 'jumped', 'over', 'Python', 'who', 'was', \"n't\", 'that', 'large', ',', 'just', '3', 'feet', 'long.', '<', '/p', '>']\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  <p>The circus dog in a plissé skirt jumped over Python who was not that large, just 3 feet long.</p>\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  [('<', 'a'), ('p', 'n'), ('>', 'v'), ('the', None), ('circus', 'n'), ('dog', 'n'), ('in', None), ('a', None), ('plissé', 'n'), ('skirt', 'n'), ('jumped', 'v'), ('over', None), ('python', 'n'), ('who', None), ('was', 'v'), (\"n't\", 'r'), ('that', None), ('large', 'a'), (',', None), ('just', 'r'), ('3', None), ('feet', 'n'), ('long.', 'a'), ('<', 'n'), ('/p', 'n'), ('>', 'n')]\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  < p > the circus dog in a plissé skirt jump over python who be n't that large , just 3 foot long. < /p >\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:    p   The circus dog in a plissé skirt jumped over Python who was n t that large   just 3 feet long     p  \n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  < p > The circus dog plissé skirt jumped Python n't large , 3 feet long. < /p >\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  p The circus dog in a plissé skirt jumped over Python who was n't that large just feet long. /p\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  <p>The circus dog in a plisse skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n"
     ]
    }
   ],
   "source": [
    "#packages needed\n",
    "\n",
    "#ignore warnings about future changes in functions as they take too much space\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning)\n",
    "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)\n",
    "\n",
    "#the module 'sys' allows istalling module from inside Jupyter\n",
    "import sys\n",
    "\n",
    "!{sys.executable} -m pip install numpy\n",
    "import numpy as np \n",
    "\n",
    "!{sys.executable} -m pip install pandas\n",
    "import pandas as pd\n",
    "\n",
    "#Natrual Language ToolKit (NLTK)\n",
    "!{sys.executable} -m pip install nltk\n",
    "import nltk\n",
    "\n",
    "#text normalization function\n",
    "%run \"C:/Users/drrr8/OneDrive - Washington University in St. Louis/Desktop/562 text mining/Text_Normalization_Function.ipynb\"\n",
    "\n",
    "#ignore warnings about future changes in functions as they take too much space\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning)\n",
    "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9b25351a",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_reviews = np.array(test_data['Review'])\n",
    "test_rate = np.array(test_data['Rating'])\n",
    "\n",
    "train_reviews = np.array(train_data['Review'])\n",
    "train_rate = np.array(train_data['Rating'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "82f1fff0",
   "metadata": {},
   "outputs": [],
   "source": [
    "normalized_test_reviews = normalize_corpus(test_reviews)\n",
    "normalized_train_reviews = normalize_corpus(train_reviews)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "37b870d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package vader_lexicon to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package vader_lexicon is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#VADER Lexicon\n",
    "nltk.download('vader_lexicon')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0b422ff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
    "analyzer = SentimentIntensityAnalyzer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "eaa9a92e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#just observe examples\n",
    "text_1 = 'nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows, not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway, maybe just noisy neighbors, aveda bath products nice, did not goldfish stay nice touch taken advantage staying longer, location great walking distance shopping, overall nice experience having pay 40 parking night,'\n",
    "text_2 = 'ok nothing special charge diamond member hilton decided chain shot 20th anniversary seattle, start booked suite paid extra website description not, suite bedroom bathroom standard hotel room, took printed reservation desk showed said things like tv couch ect desk clerk told oh mixed suites description kimpton website sorry free breakfast, got kidding, embassy suits sitting room bathroom bedroom unlike kimpton calls suite, 5 day stay offer correct false advertising, send kimpton preferred guest website email asking failure provide suite advertised website reservation description furnished hard copy reservation printout website desk manager duty did not reply solution, send email trip guest survey did not follow email mail, guess tell concerned guest.the staff ranged indifferent not helpful, asked desk good breakfast spots neighborhood hood told no hotels, gee best breakfast spots seattle 1/2 block away convenient hotel does not know exist, arrived late night 11 pm inside run bellman busy chating cell phone help bags.prior arrival emailed hotel inform 20th anniversary half really picky wanted make sure good, got nice email saying like deliver bottle champagne chocolate covered strawberries room arrival celebrate, told needed foam pillows, arrival no champagne strawberries no foam pillows great room view alley high rise building good not better housekeeping staff cleaner room property, impressed left morning shopping room got short trips 2 hours, beds comfortable.not good ac-heat control 4 x 4 inch screen bring green shine directly eyes night, light sensitive tape controls.this not 4 start hotel clean business hotel super high rates, better chain hotels seattle'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f10037b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice previous reviews did valet parking, check quick easy, little disappointed non-existent view room room clean nice size, bed comfortable woke stiff neck high pillows, not soundproof like heard music room night morning loud bangs doors opening closing hear people talking hallway, maybe just noisy neighbors, aveda bath products nice, did not goldfish stay nice touch taken advantage staying longer, location great walking distance shopping, overall nice experience having pay 40 parking night, \n",
      "Scores: {'neg': 0.072, 'neu': 0.643, 'pos': 0.285, 'compound': 0.9747} \n",
      "\n",
      "ok nothing special charge diamond member hilton decided chain shot 20th anniversary seattle, start booked suite paid extra website description not, suite bedroom bathroom standard hotel room, took printed reservation desk showed said things like tv couch ect desk clerk told oh mixed suites description kimpton website sorry free breakfast, got kidding, embassy suits sitting room bathroom bedroom unlike kimpton calls suite, 5 day stay offer correct false advertising, send kimpton preferred guest website email asking failure provide suite advertised website reservation description furnished hard copy reservation printout website desk manager duty did not reply solution, send email trip guest survey did not follow email mail, guess tell concerned guest.the staff ranged indifferent not helpful, asked desk good breakfast spots neighborhood hood told no hotels, gee best breakfast spots seattle 1/2 block away convenient hotel does not know exist, arrived late night 11 pm inside run bellman busy chating cell phone help bags.prior arrival emailed hotel inform 20th anniversary half really picky wanted make sure good, got nice email saying like deliver bottle champagne chocolate covered strawberries room arrival celebrate, told needed foam pillows, arrival no champagne strawberries no foam pillows great room view alley high rise building good not better housekeeping staff cleaner room property, impressed left morning shopping room got short trips 2 hours, beds comfortable.not good ac-heat control 4 x 4 inch screen bring green shine directly eyes night, light sensitive tape controls.this not 4 start hotel clean business hotel super high rates, better chain hotels seattle \n",
      "Scores: {'neg': 0.11, 'neu': 0.701, 'pos': 0.189, 'compound': 0.9787} \n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(text_1, \"\\nScores:\", analyzer.polarity_scores(text_1),\"\\n\")\n",
    "print(text_2, \"\\nScores:\", analyzer.polarity_scores(text_2),\"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "970fdbc0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Review:-\n",
      "fantastic stayed excellence punta cana february vacation realize days n't, resort fantastic, clean staff friendly drinks plentiful, highly recommend coco loco served coconut, overall food good consider picky eater, lobster place favorite, spent time sitting pool ocean bit rough taste lounging inside raining, did n't advantage horseback riding bike tour heard fun, recommend leaving property going plaza time, shops fun souvenirs haggle vendors lot fun, downside leaving property taxi cost 35 way, better idea country like taxi ride though.my complaint n't night life, michael jackson best, disco fun especially staff hand dancing dance lessons afternoons, went vacation friend arrived obvious resort catered mainly couples not advertized website, hard time getting room beds nicolas extremely helpful situation, emilio concierge friendly informative entertainment crew isael best,  \n",
      "Actual Rate: 4\n",
      "\n",
      "Review:-\n",
      "loved, ca n't wait come, wonderful stay, super friendly folks beautiful room extremely convenient location, breakfast filling american sense not super fancy free filling used eating hearty breakfast morning, super sweet allowed bags past outr stay hop skip jump train station, bofriend just recently went business upgraded free larger bed, maitre gem truly felt staying grand place, recommend wonder restaurant near called palo oro literally golden balls good wonderful food,  \n",
      "Actual Rate: 5\n",
      "\n"
     ]
    }
   ],
   "source": [
    "sample_docs = [100, 700] #indecies for 2 sample documnets \n",
    "\n",
    "for doc_index in sample_docs:\n",
    "    print('Review:-')\n",
    "    print(test_reviews[doc_index])\n",
    "    print('Actual Rate:', test_rate[doc_index])\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "09189195",
   "metadata": {},
   "outputs": [],
   "source": [
    "#function that scores text using VADER lexicon and prints the actual and scored sentiment\n",
    "\n",
    "def analyze_sentiment_vader_lexicon(review, threshold = 0.1, verbose = False):\n",
    "    scores = analyzer.polarity_scores(review)  \n",
    "    binary_sentiment = 'positive' if scores['compound'] >= threshold else 'negative'\n",
    "    if verbose:                             \n",
    "        print('VADER Polarity (Binary):', binary_sentiment)\n",
    "        print('VADER Score:', round(scores['compound'], 2))\n",
    "    return binary_sentiment,scores['compound'] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "cb7e793a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Review text:\n",
      "\n",
      "fantastic stayed excellence punta cana february vacation realize days n't, resort fantastic, clean staff friendly drinks plentiful, highly recommend coco loco served coconut, overall food good consider picky eater, lobster place favorite, spent time sitting pool ocean bit rough taste lounging inside raining, did n't advantage horseback riding bike tour heard fun, recommend leaving property going plaza time, shops fun souvenirs haggle vendors lot fun, downside leaving property taxi cost 35 way, better idea country like taxi ride though.my complaint n't night life, michael jackson best, disco fun especially staff hand dancing dance lessons afternoons, went vacation friend arrived obvious resort catered mainly couples not advertized website, hard time getting room beds nicolas extremely helpful situation, emilio concierge friendly informative entertainment crew isael best,   \n",
      "\n",
      "ACTUAL Rate : 4 \n",
      "\n",
      "VADER Polarity (Binary): positive\n",
      "VADER Score: 1.0\n",
      "------------------------------------------------------------\n",
      "\n",
      "Review text:\n",
      "\n",
      "loved, ca n't wait come, wonderful stay, super friendly folks beautiful room extremely convenient location, breakfast filling american sense not super fancy free filling used eating hearty breakfast morning, super sweet allowed bags past outr stay hop skip jump train station, bofriend just recently went business upgraded free larger bed, maitre gem truly felt staying grand place, recommend wonder restaurant near called palo oro literally golden balls good wonderful food,   \n",
      "\n",
      "ACTUAL Rate : 5 \n",
      "\n",
      "VADER Polarity (Binary): positive\n",
      "VADER Score: 0.99\n",
      "------------------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "for doc_index in sample_docs:\n",
    "    print('\\nReview text:\\n')\n",
    "    print(test_reviews[doc_index],\"\\n\")\n",
    "    print('ACTUAL Rate :', test_rate[doc_index],\"\\n\")    \n",
    "    final_sentiment = analyze_sentiment_vader_lexicon(normalized_test_reviews[doc_index],\n",
    "                                                        threshold=0.1,\n",
    "                                                        verbose=True)\n",
    "    print('-'*60) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a54a0b49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>VADER Polarity</th>\n",
       "      <th>VADER Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>negative</td>\n",
       "      <td>-0.9481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>positive</td>\n",
       "      <td>0.9891</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>positive</td>\n",
       "      <td>0.9690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>negative</td>\n",
       "      <td>-0.8286</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>positive</td>\n",
       "      <td>0.9943</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  VADER Polarity  VADER Score\n",
       "0       negative      -0.9481\n",
       "1       positive       0.9891\n",
       "2       positive       0.9690\n",
       "3       negative      -0.8286\n",
       "4       positive       0.9943"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#score  all hotel reviews in the test dataset:\n",
    "VADER_polarity_test = [analyze_sentiment_vader_lexicon(review, threshold=0.1) for review in test_reviews]\n",
    "VADER_polarity_test_df = pd.DataFrame(VADER_polarity_test, columns = ['VADER Polarity','VADER Score'])\n",
    "VADER_polarity_test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "acf7a78c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>VADER Polarity</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>positive</th>\n",
       "      <td>5631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>negative</th>\n",
       "      <td>517</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Total</th>\n",
       "      <td>6148</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                count\n",
       "VADER Polarity       \n",
       "positive         5631\n",
       "negative          517\n",
       "Total            6148"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "polarity_count = VADER_polarity_test_df['VADER Polarity'].value_counts()\n",
    "polarity_count_df = pd.DataFrame(polarity_count).rename(columns={'VADER Polarity': 'Count'})\n",
    "total_count = len(VADER_polarity_test_df)\n",
    "polarity_count_df.loc['Total'] = total_count\n",
    "polarity_count_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "48eb8cf5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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dMjgsm+88QZcXFzQvHlz7Nu3T2tedHS0TputWrUy2GZKSgqSk5O1JiIiIno9lZiwdPfuXWRkZMDJyUmr3MnJCQkJCXqXcXFxwffff49169Zh/fr1qFy5Mpo3b44DBw4odRISEvLVJgCEhoZCo9Eok5ub20tsGRERERVnJkXdgfxSqVRar0VEpyxL5cqVUblyZeW1r68vrl27hq+//hpNmjT5V20CQEhICEaOHKm8Tk5OZmAiIiJ6TZWYI0v29vYwNjbWOeJz+/ZtnSNDhjRs2BAXL15UXjs7O+e7TbVaDRsbG62JiIiouPAYs5VjlwpQiQlLZmZm8PHxQWRkpFZ5ZGQk/Pz88tzOyZMn4eLiorz29fXVaXPXrl35apOIiIheXyXqNNzIkSMRFBSEunXrwtfXF99//z3i4uIwcOBAAM9Pj924cQPLly8H8PxKNw8PD1SrVg2pqalYuXIl1q1bh3Xr1iltDh8+HE2aNMGMGTPQvn17bNq0Cbt378Zvv/1WJNtIRERExUuJCkudO3fGvXv3MGXKFMTHx6N69erYtm0b3N3dAQDx8fFa91xKTU3FJ598ghs3bsDCwgLVqlXD1q1b0aZNG6WOn58f1qxZg88//xzjx49HhQoVEBERgQYNGrzy7SMiIipIHmO24sr0tkXdjRKvRIUlABg0aBAGDRqkd154eLjW69GjR2P06NG5ttmhQwd06NChILpHREREr5kSM2aJiIiIqCgwLBEREREZwLBERET0GuMtBF4ewxIREdFrjoHp5TAsERERERnAsERERERkAMMSERERkQEMS0REREQGMCwRERERGcCwRERERGQAwxIRERGRAQxLRERERAYwLBEREREZwLBEREREZADDEhEREZEBDEtERET/IXxOXP4xLBEREREZwLBEREREZADDEhEREZEBDEtEREREBjAsERERERnAsERERERkAMMSERERkQEMS0REREQGMCwRERERGcCwRERE9B/Du3jnD8MSERERkQEMS0RERP9BPLqUdwxLRERERAYwLBEREREZwLBEREREZECJC0sLFy6Ep6cnzM3N4ePjg4MHD+ZYd/369WjZsiUcHBxgY2MDX19f7Ny5U6tOeHg4VCqVzvTs2bPC3hQiIiIqAUpUWIqIiEBwcDDGjRuHkydPonHjxmjdujXi4uL01j9w4ABatmyJbdu24fjx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l1GpVHqnr776Sqnj7++vM79Lly6FvDVERESvl9d5kLdJUXcgr7p164br169jx44dAIAPP/wQQUFB2LJlS47LxMfHa73evn07+vXrhw8++ECrfMCAAZgyZYry2sLCogB7TkRERCVZiQhL586dw44dOxATE4MGDRoAAH744Qf4+vri/PnzqFy5st7lnJ2dtV5v2rQJzZo1g5eXl1a5paWlTl0iIiIioISchouOjoZGo1GCEgA0bNgQGo0Ghw4dylMbt27dwtatW9GvXz+deatWrYK9vT2qVauGTz75BA8fPjTYVkpKCpKTk7UmIiKi/7rX9VRciTiylJCQAEdHR51yR0dHJCQk5KmNZcuWwdraGu+//75Weffu3eHp6QlnZ2ecOXMGISEhOHXqFCIjI3NsKzQ0FJMnT87fRhAREVGJVKRHliZNmpTjIOys6dixYwCeD9bOTkT0luuzdOlSdO/eHebm5lrlAwYMQIsWLVC9enV06dIFv/zyC3bv3o0TJ07k2FZISAiSkpKU6dq1a/nYaiIiIipJivTI0pAhQ3K98szDwwN//PEHbt26pTPvzp07cHJyynU9Bw8exPnz5xEREZFr3Tp16sDU1BQXL15EnTp19NZRq9VQq9W5tkVEREQlX5GGJXt7e9jb2+daz9fXF0lJSThy5Ajq168PADh8+DCSkpLg5+eX6/JLliyBj48PatWqlWvds2fPIi0tDS4uLrlvABEREWl5HcctlYgB3t7e3nj77bcxYMAAxMTEICYmBgMGDEBgYKDWlXBVqlTBhg0btJZNTk7G2rVr0b9/f512//nnH0yZMgXHjh3DlStXsG3bNnTs2BFvvPEGGjVqVOjbRURERMVfiQhLwPMr1mrUqIGAgAAEBASgZs2aWLFihVad8+fPIykpSatszZo1EBF07dpVp00zMzPs2bMHrVq1QuXKlTFs2DAEBARg9+7dMDY2LtTtISIiopKhRFwNBwBlypTBypUrDdYREZ2yDz/8EB9++KHe+m5uboiKiiqQ/hEREdHrqcQcWSIiIiIqCgxLRERERAYwLBEREREZwLBEREREZADDEhEREZEBDEtEREREBjAsERERERnAsERERERkAMMSERERkQEMS0REREQGMCwRERERGcCwRERERIXiyvS2uDK9bVF346UxLBEREREZwLBEREREhaqkH2FiWCIiIqJX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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "pos_reviews_scored = VADER_polarity_test_df[:500]['VADER Score']\n",
    "plt.bar(range(0, 500), pos_reviews_scored.sort_values(ascending=False))\n",
    "plt.xlabel(\"Reviews with positive ACTUAL polarity\")\n",
    "plt.ylabel(\"VADER Score\")\n",
    "plt.title(\"VADER Scores for ACTUAL Positive Reviews \\n (in decreasing order of VADER scores)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6723ca47",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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VCk0a9YMu3fvVo6fPn0aKpUK69evV9KOHDmiTMc+q0uXLqhfv76yv2fPHvj6+sLOzg4WFhaoUKEC3n77bTx8+DCfPfTfxmCJXtjgwYOhUql0vsDOnDmDP//8EwMGDICxsbGSvnTpUqjVavTt21c5V98HTG66dOkCEcH+/fu10kUEGRkZOltevqguXboEAKhatWqueb29vREaGopp06bh+PHjeteNZBs4cCA++OADNGzYEOvWrcPatWvRpUsXrQ/L4cOHY8KECWjTpg22bNmCzz77DGFhYWjatClu3bqlVV5CQgL69euHPn36YPv27RgxYgQePnwIHx8fLF++HKNHj8aOHTswYcIEhISEKH0FANHR0ejVqxcqVaqEtWvXYtu2bfj000+RkZGRY/2dnZ0RHR2NunXrolKlSoiOjkZ0dDTq1auHx48fw8/PDytWrMDYsWOxbds29OvXD3PnzkX37t11ytq0aRMWLVqETz/9FDt37kTz5s0N9vOGDRtw9+5dDB48GFWqVMGbb76JdevW4f79+0qew4cP486dO+jatWuRrH953rVr17Br1y507doVZcuWxYABA/DXX39h3759BX4tb29vvPXWW5gzZw7u3LmTY768vj+ysrLQuXNn/PLLL5gwYQI2btyIxo0b650qu3HjBuzs7DB79myEhYXh+++/h4mJCRo3boz4+HgAT9eaBQcHAwA++eQT5b3z7rvv6q3n4MGDceXKFezZs0cr/dy5c/jzzz8xaNAgJe29997DmDFj0Lp1a2zatAk//PADTp8+jaZNm+bpPzz66PsMiIyMRLNmzZCcnIzFixdj8+bNqFOnDnr16qWsV/Py8kLdunWVtj4rJCQEDg4O6NChQ47XXbVqFfz9/WFtbY3ly5fj119/ha2tLdq2basETDVr1oSzszMiIiKU8yIiImBhYYEzZ87gxo0bAJ4ub4iKikLr1q0BPP3PTseOHWFmZoZly5YhLCwMs2fPhqWlJZ48efJC/fSfVYSjWvQa8PHxEXt7e3ny5ImSNm7cOAEg58+fV9IuX74sRkZG0rt3b61zLS0tJTU1VafMnKbhRER27NghAGTOnDlKmpubmwDQu3322WdKvuwh+JiYGElPT5d79+5JWFiYODk5SYsWLSQ9PT3XNv/111/i6emplG9hYSGtWrWS7777Tqsf9u3bJwBk8uTJOZZ1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qlRBamoqvv32W8TFxeH7779/JW0iIiKi4q1EBUu9evXC7du3MWPGDCQkJMDT0xPbt29X7m5LSEjQeeZSSkoKQkND8c033+gtMzk5GUOHDkViYiI0Gg3q1q2Lffv2oVGjRoXeHsofTsEREVFRKFHBEgCMGDECI0aM0HssJCREJ02j0eDhw4c5ljd//nzMnz+/oKpHhYSBEhERFZUSs2aJiIiIqCgwWCIiIiIygMESFXucgiMioqLEYImIiIjIAAZLRERERAYwWCIiIiIygMESERERkQEMlqhY4qJuIiIqLhgsUbHDQImIiIoTBktEREREBjBYomKFo0pERFTcMFgiIiIiMoDBEhUbHFUiIqLiiMESERERkQEMlqhY4KgSEREVVwyWiIiIiAxgsERERERkAIMlIiIiIgMYLFGR43olIiIqzhgsUZFioERERMUdgyUiIiIiAxgsUZHhqBIREZUEDJaIiIiIDGCwRERERGQAgyUiIiIiAxgsERERERnAYImIiIjIAAZLRERERAYwWCIiIiIygMESERERkQEMloiIiIgMYLBEREREZECJC5Z++OEHuLu7w9zcHPXr18f+/ftzzLt3716oVCqd7dy5c1r5QkNDUaNGDajVatSoUQMbN24s7GYQERFRCVGigqV169ZhzJgxmDx5Mo4dO4bmzZujffv2uHLlisHz4uPjkZCQoGxVqlRRjkVHR6NXr14IDAzE8ePHERgYiJ49e+LQoUOF3RwiIiIqAUpUsDRv3jwEBQXh3XffRfXq1bFgwQK4urpi0aJFBs9zcHCAk5OTshkbGyvHFixYgDZt2mDSpEnw8PDApEmT0KpVKyxYsKCQW0NEREQlQYkJlp48eYIjR47A399fK93f3x8HDx40eG7dunXh7OyMVq1aITIyUutYdHS0Tplt27Y1WGZaWhpSU1O1NiIiIno9lZhg6datW8jMzISjo6NWuqOjIxITE/We4+zsjJ9++gmhoaHYsGEDqlWrhlatWmHfvn1KnsTExHyVCQCzZs2CRqNRNldX15doGRERERVnJkVdgfxSqVRa+yKik5atWrVqqFatmrLv7e2Nq1ev4quvvkKLFi1eqEwAmDRpEsaOHavsp6amMmAiIiJ6TZWYkSV7e3sYGxvrjPgkJSXpjAwZ0qRJE1y4cEHZd3JyyneZarUa1tbWWhvlruLEbag4cVtRV4OIiChfSkywZGZmhvr16yM8PFwrPTw8HE2bNs1zOceOHYOzs7Oy7+3trVPmrl278lUmERERvb5K1DTc2LFjERgYiAYNGsDb2xs//fQTrly5gmHDhgF4Oj12/fp1rFixAsDTO90qVqyImjVr4smTJ1i1ahVCQ0MRGhqqlPnBBx+gRYsWmDNnDrp27YrNmzcjIiICf/zxR5G0kYiIiIqXEhUs9erVC7dv38aMGTOQkJAAT09PbN++HW5ubgCAhIQErWcuPXnyBOPHj8f169dhYWGBmjVrYtu2bejQoYOSp2nTpli7di0++eQTTJkyBZUrV8a6devQuHHjV96+1xmn34iIqKQqUcESAIwYMQIjRozQeywkJERr/6OPPsJHH32Ua5kBAQEICAgoiOoRERHRa6bErFkiIiIiKgoMloiIiIgMYLBEhY7rlYiIqCRjsESFioESERGVdAyWiIiIiAxgsERERERkAIMlKjScgiMiotcBgyUiIiIiAxgsERERERnAYIkKBafgiIjodcFgiYiIiMgABktEREREBjBYIiIiIjKAwRIRERGRAQyWiIiIiAxgsERERERkAIMlIiIiIgMYLBEREREZwGCJiIiIyAAGS0REREQGMFgiIiIiMoDBEhEREZEBDJaIiIiIDGCwRERERGQAgyUiIiIiAxgsERERERnAYImIiIjIAAZLVGAqTtxW1FUgIiIqcAyWiIiIiAxgsERERERkAIMlIiIiIgNKXLD0ww8/wN3dHebm5qhfvz7279+fY94NGzagTZs2KFu2LKytreHt7Y2dO3dq5QkJCYFKpdLZHj9+XNhNISIiohKgRAVL69atw5gxYzB58mQcO3YMzZs3R/v27XHlyhW9+fft24c2bdpg+/btOHLkCPz8/NC5c2ccO3ZMK5+1tTUSEhK0NnNz81fRJCIiIirmTF70xL/++gsXL15EixYtYGFhARGBSqUqyLrpmDdvHoKCgvDuu+8CABYsWICdO3di0aJFmDVrlk7+BQsWaO3PnDkTmzdvxtatW1G3bl0lXaVSwcnJqVDrTkRERCVTvkeWbt++jdatW6Nq1aro0KEDEhISAADvvvsuxo0bV+AVzPbkyRMcOXIE/v7+Wun+/v44ePBgnsrIysrCvXv3YGtrq5V+//59uLm5oXz58ujUqZPOyNPz0tLSkJqaqrURERHR6ynfwdKHH34IExMTXLlyBaVKlVLSe/XqhbCwsAKt3LNu3bqFzMxMODo6aqU7OjoiMTExT2V8/fXXePDgAXr27KmkeXh4ICQkBFu2bMGaNWtgbm6OZs2a4cKFCzmWM2vWLGg0GmVzdXV9sUYRERFRsZfvabhdu3Zh586dKF++vFZ6lSpV8M8//xRYxXLy/FRfXqf/1qxZg2nTpmHz5s1wcHBQ0ps0aYImTZoo+82aNUO9evWwcOFCfPvtt3rLmjRpEsaOHavsp6amMmAiIiJ6TeU7WHrw4IHWiFK2W7duQa1WF0il9LG3t4exsbHOKFJSUpLOaNPz1q1bh6CgIKxfvx6tW7c2mNfIyAgNGzY0OLKkVqsLta1ERERUfOR7Gq5FixZYsWKFsq9SqZCVlYUvv/wSfn5+BVq5Z5mZmaF+/foIDw/XSg8PD0fTpk1zPG/NmjUYOHAgfvnlF3Ts2DHX64gI4uLi4Ozs/NJ1JiIiopIv3yNLX375JXx9fXH48GE8efIEH330EU6fPo07d+7gwIEDhVFHxdixYxEYGIgGDRrA29sbP/30E65cuYJhw4YBeDo9dv36dSWYW7NmDfr3749vvvkGTZo0UUalLCwsoNFoAADTp09HkyZNUKVKFaSmpuLbb79FXFwcvv/++0JtCxEREZUM+Q6WatSogRMnTmDRokUwNjbGgwcP0L17d4wcObLQR2N69eqF27dvY8aMGUhISICnpye2b98ONzc3AEBCQoLWM5d+/PFHZGRkYOTIkRg5cqSSPmDAAISEhAAAkpOTMXToUCQmJkKj0aBu3brYt28fGjVqVKhtISIiopIhX8FSeno6/P398eOPP2L69OmFVSeDRowYgREjRug9lh0AZdu7d2+u5c2fPx/z588vgJoRERHR6yhfa5ZMTU1x6tSpQn/4JBEREVFxke8F3v3798fSpUsLoy5ERERExU6+1yw9efIEP//8M8LDw9GgQQNYWlpqHZ83b16BVY6IiIioqOU7WDp16hTq1asHADh//rzWMU7PERER0esm38FSZGRkYdSDiIiIqFjK95qlZ127dg3Xr18vqLoQERERFTv5DpaysrIwY8YMaDQauLm5oUKFCihTpgw+++wzZGVlFUYdiYiIiIpMvqfhJk+ejKVLl2L27Nlo1qwZRAQHDhzAtGnT8PjxY3zxxReFUU8iIiKiIpHvYGn58uX4+eef0aVLFyXNy8sL5cqVw4gRIxgsERER0Wsl39Nwd+7cgYeHh066h4cH7ty5UyCVopKn4sRtRV0FIiKiQpHvYMnLywvfffedTvp3330HLy+vAqkUERERUXGR72m4uXPnomPHjoiIiIC3tzdUKhUOHjyIq1evYvv27YVRRyIiIqIik++RJR8fH8THx+Ott95CcnIy7ty5g+7duyM+Ph7NmzcvjDoSERERFZl8jywBQLly5biQm4iIiP4T8j2yFBwcjPXr1+ukr1+/HsuXLy+QShEREREVF/kOlmbPng17e3uddAcHB8ycObNAKkVERERUXOQ7WPrnn3/g7u6uk+7m5oYrV64USKWIiIiIiot8B0sODg44ceKETvrx48dhZ2dXIJUiIiIiKi7yHSz17t0bo0ePRmRkJDIzM5GZmYk9e/bggw8+QO/evQujjkRERERFJt93w33++ef4559/0KpVK5iYPD09KysL/fv355olIiIieu3kO1gyMzPDunXr8PnnnyMuLg4WFhaoVasW3NzcCqN+REREREXqhZ6zBABVqlRBlSpVkJmZiZMnT8La2ho2NjYFWTciIiKiIpfvNUtjxozB0qVLAQCZmZnw8fFBvXr14Orqir179xZ0/YiIiIiKVL6Dpd9++035wdytW7fi77//xrlz5zBmzBhMnjy5wCtIREREVJTyHSzdunULTk5OAIDt27ejZ8+eqFq1KoKCgnDy5MkCryARERFRUcp3sOTo6IgzZ84gMzMTYWFhaN26NQDg4cOHMDY2LvAKUvFXceK2oq4CERFRocn3Au9BgwahZ8+ecHZ2hkqlQps2bQAAhw4dgoeHR4FXkIiIiKgo5TtYmjZtGjw9PXH16lX06NEDarUaAGBsbIyJEycWeAWJiIiIitILPTogICBAJ23AgAEvXRkiIiKi4ibfa5aIiIiI/ksYLBEREREZUOKCpR9++AHu7u4wNzdH/fr1sX//foP5o6KiUL9+fZibm6NSpUpYvHixTp7Q0FDUqFEDarUaNWrUwMaNGwur+kRERFTClKhgad26dcrDL48dO4bmzZujffv2uHLlit78ly5dQocOHdC8eXMcO3YMH3/8MUaPHo3Q0FAlT3R0NHr16oXAwEAcP34cgYGB6NmzJw4dOvSqmkVERETFWIEFS0ePHkWnTp0Kqji95s2bh6CgILz77ruoXr06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asHf3x/+/v6oXbs2Vq5cqZUnPj5e+d26a9euYcuWLbh27Rrq1KkDZ2dnZcu+g87MzAy7d+9G27ZtUa1aNYwePRr+/v6IiIgosHVZREREVLIV6TRcftja2mLVqlUG84iI8u+KFStq7evj6uqKqKioAqkfERERvZ5KzMgSERERUVFgsERERERkAIMlIiIiIgMYLBEREREZwGCJiIiIyAAGS0REREQGMFgiIiIiMoDBEhEREZEBDJaIiIiIDGCwRERERGQAgyUiIiIiAxgsERERERnAYImIiIjIAAZLRERERAYwWKJcXZ7dsairQEREVGQYLBEREREZwGCJiIiIyAAGS0REREQGMFgiIiIiMoDBEhEREZEBDJaIiIiIDGCwRERERGQAgyUyiM9YIiKi/zoGS5QjBkpEREQMloiIiIgMYrBEREREZACDJSIiIiIDGCwRERERGcBgiYiIiMgABktEREREBjBYIiIiIjKgxARLd+/eRWBgIDQaDTQaDQIDA5GcnGzwnIEDB0KlUmltTZo00cqTlpaG999/H/b29rC0tESXLl1w7dq1QmwJERERlSQlJljq06cP4uLiEBYWhrCwMMTFxSEwMDDX89q1a4eEhARl2759u9bxMWPGYOPGjVi7di3++OMP3L9/H506dUJmZmZhNYWIiIhKEJOirkBenD17FmFhYYiJiUHjxo0BAEuWLIG3tzfi4+NRrVq1HM9Vq9VwcnLSeywlJQVLly7FypUr0bp1awDAqlWr4OrqioiICLRt27bgG0NEREQlSokYWYqOjoZGo1ECJQBo0qQJNBoNDh48aPDcvXv3wsHBAVWrVsWQIUOQlJSkHDty5AjS09Ph7++vpLm4uMDT09NguWlpaUhNTdXaXieXZ3fkT50QERH9fyUiWEpMTISDg4NOuoODAxITE3M8r3379li9ejX27NmDr7/+GrGxsWjZsiXS0tKUcs3MzGBjY6N1nqOjo8FyZ82apayd0mg0cHV1fcGWFT8MkoiIiLQVabA0bdo0nQXYz2+HDx8GAKhUKp3zRURverZevXqhY8eO8PT0ROfOnbFjxw6cP38e27ZtM1iv3MqdNGkSUlJSlO3q1at5bDERERGVNEW6ZmnUqFHo3bu3wTwVK1bEiRMn8O+//+ocu3nzJhwdHfN8PWdnZ7i5ueHChQsAACcnJzx58gR3797VGl1K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DQ0Fhjmld+7cGa6urliyZAlcXFyQlZUFT09Pndf32QAsP/r374+JEyciOjoa0dHRqFixIpo3b/5CZVHJw5ElohxMnToVX331FW7cuAFHR0eUK1cOf//9N9544w2tLft/sADQrVs3PH78GGFhYfjll1+0/kf+vHr16iExMREmJiY6Zdrb2wMANBoN6tSpg++++w4qlQo1atRA8+bNcezYMfz+++9ai4qBp1+CvXr1wpIlS7Bu3TqEhobizp07eq/v6ekJOzs7fP755/Dy8oK1tTV8fHwQFRWFvXv36pT9LDMzM2XEo7DlpZ+qVaum3P2V7fDhw1r7+/fvx+jRo9GhQwfUrFkTarVaWeCezdTUNE/t6tOnD1avXo2tW7fCyMgIHTt21Krv6dOnder6xhtvwMzMTG95S5cuRe/evREXF6e19e3bV1lQ3adPH9y/f1/rpoJn5fYMrGd5eHjovTMuNjYW1apV06rXuHHjtOp0/Phx+Pn5vfToUkxMjM5+9qMJatSogbi4OK1F8QcOHICRkRGqVq2qU9bt27dx9uxZfPLJJ2jVqhWqV6/+wo/9yOm9bWdnh27duiE4OBjBwcEYNGjQC5VPJVTRzgISFQ85rX+oX7++shZmyZIlYmFhIQsWLJD4+Hg5ceKELFu2TL7++mutc/r06SNeXl6iUqnkn3/+0Tr2/ALvN998U7y8vCQsLEwuXbokBw4ckMmTJ0tsbKxyztixY8XY2FgCAgKUtDp16oixsbF8//33Stq8efNkzZo1cvbsWYmPj5egoCBxcnJSFrjq0717dzE2Npbx48crdbK1tRVjY2NlXZSI7pqlAwcOCACJiIiQmzdvyoMHD3Tal83Ly0umTp2aYx18fHzkgw8+0Err2rWrspg3L/20atUqsbCwkJCQEDl//rx89tlnYm1tLXXq1NHqszZt2siZM2ckJiZGmjdvLhYWFlr1rVKligwfPlwSEhKUNUj61rCcP39eAEjt2rUlKChI69jx48fFwsJCRowYIceOHZPz58/L5s2blfVhz0tKShJTU1Nlfdizdu3aJaampspi848++kiMjY3lf//7nxw8eFAuX74sEREREhAQoLMWx9CapejoaDEyMpLp06fL6dOn5fTp0zJjxgwxMjKSmJgYERE5duyYANC7OP+nn36SsmXLKovN8f/X9x07dkxry2nxMwCxt7eXpUuXSnx8vHz66adiZGQkp0+fFpGnC7ydnZ3l7bfflpMnT8qePXukUqVKOS7wzszMFDs7O+nXr59cuHBBdu/eLQ0bNtRaS5VTfzy/Zmn16tViaWkpx44dk5s3b2qtldu1a5eYmZmJsbGxXL9+XW/b6PXEYIlIcg6WVq9eLWZmZnLlyhVlv06dOmJmZiY2NjbSokUL2bBhg9Y527ZtEwDSokULnfKeDyZSU1Pl/fffFxcXFzE1NRVXV1fp27evcj0Rka1btwoA+e6775S0Dz74QADIqVOnlLSffvpJ6tSpI5aWlmJtbS2tWrWSo0ePGmz3woULBYD8/vvvSlrXrl3F2NhYueNORDdYEhEZNmyY2NnZCQAlGCqMYEkkb/00Y8YMsbe3l9KlS8vgwYNl9OjR0qRJE+X40aNHpUGDBqJWq6VKlSqyfv16nfpu2bJF3njjDTExMVEWOue04Df7y3jPnj06x/78809p06aNlC5dWiwtLaV27dryxRdf6G3/V199JWXKlNF7l1t6errY2tpqBeTr1q2TFi1aiJWVlVL2jBkztF4bEcPBkohIeHi4NG/eXGxsbMTGxkbefPNNCQ8PV46PGjVKatSooffcpKQkMTY2ltDQUBER5YaI57fIyEi95wOQ77//Xtq0aSNqtVrc3NxkzZo1WnlOnDghfn5+Ym5uLra2tjJkyBCt4Ov5v9nw8HCpXr26qNVqqV27tuzdu/eFgqXHjx/L22+/LWXKlFHujsyWlZUlbm5u0qFDB73toteXSuS5iX0iotdAmzZt4OTkhJUrVxZ1Veg5KpUKGzduLJCfK3mVHj58CBcXFyxbtgzdu3cv6urQK8QF3kRU4j18+BCLFy9G27ZtYWxsjDVr1iAiIgLh4eFFXTV6DWRlZSExMRFff/01NBoNunTpUtRVoleMwRIRlXgqlQrbt2/H559/jrS0NFSrVg2hoaFo3bp1UVeNXgNXrlyBu7s7ypcvj5CQEJiY8Kvzv4bTcEREREQG8NEBRERERAYwWCIiIiIygMESERERkQEMloiIiIgMYLBEREREZACDJSIiIiIDGCwRERERGcBgiYiIiMiA/wcfcLk9odDfvgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "neg_reviews_scored = VADER_polarity_test_df[0:500]['VADER Score']\n",
    "plt.bar(range(0, 500), neg_reviews_scored.sort_values(ascending=True))\n",
    "plt.xlabel(\"Reviews with negative ACTUAL polarity\")\n",
    "plt.ylabel(\"VADER score\")\n",
    "plt.title(\"VADER Scores for ACTUAL Negative Reviews \\n (in ascending order of VADER scores)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "7cddc662",
   "metadata": {},
   "outputs": [],
   "source": [
    "# divide into positive and negative\n",
    "positive_corpus = test_reviews[VADER_polarity_test_df['VADER Polarity'] == 'positive'].tolist()\n",
    "negative_corpus = test_reviews[VADER_polarity_test_df['VADER Polarity'] == 'negative'].tolist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "d616edf0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy in d:\\anaconda\\setup\\lib\\site-packages (1.24.3)\n",
      "Requirement already satisfied: pandas in d:\\anaconda\\setup\\lib\\site-packages (2.0.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: numpy>=1.20.3 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (1.24.3)\n",
      "Requirement already satisfied: tzdata>=2022.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: pytz>=2020.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas) (2022.1)\n",
      "Requirement already satisfied: six>=1.5 in d:\\anaconda\\setup\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: nltk in d:\\anaconda\\setup\\lib\\site-packages (3.7)\n",
      "Requirement already satisfied: tqdm in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (4.64.1)\n",
      "Requirement already satisfied: regex>=2021.8.3 in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (2022.7.9)\n",
      "Requirement already satisfied: click in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (8.0.4)\n",
      "Requirement already satisfied: joblib in d:\\anaconda\\setup\\lib\\site-packages (from nltk) (1.2.0)\n",
      "Requirement already satisfied: colorama in d:\\anaconda\\setup\\lib\\site-packages (from click->nltk) (0.4.5)\n",
      "Requirement already satisfied: sklearn in d:\\anaconda\\setup\\lib\\site-packages (0.0.post4)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: Invalid requirement: '#visualizing'\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: html.parser in d:\\anaconda\\setup\\lib\\site-packages (0.2)\n",
      "Requirement already satisfied: ply in d:\\anaconda\\setup\\lib\\site-packages (from html.parser) (3.11)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package omw-1.4 to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package omw-1.4 is already up-to-date!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pattern3 in d:\\anaconda\\setup\\lib\\site-packages (3.0.0)\n",
      "Requirement already satisfied: simplejson in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (3.19.1)\n",
      "Requirement already satisfied: docx in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (0.2.4)\n",
      "Requirement already satisfied: cherrypy in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (18.8.0)\n",
      "Requirement already satisfied: feedparser in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (6.0.10)\n",
      "Requirement already satisfied: beautifulsoup4 in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (4.11.1)\n",
      "Requirement already satisfied: pdfminer3k in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (1.3.4)\n",
      "Requirement already satisfied: pdfminer.six in d:\\anaconda\\setup\\lib\\site-packages (from pattern3) (20221105)\n",
      "Requirement already satisfied: soupsieve>1.2 in d:\\anaconda\\setup\\lib\\site-packages (from beautifulsoup4->pattern3) (2.3.2.post1)\n",
      "Requirement already satisfied: more-itertools in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (9.1.0)\n",
      "Requirement already satisfied: zc.lockfile in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (3.0.post1)\n",
      "Requirement already satisfied: portend>=2.1.1 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (3.1.0)\n",
      "Requirement already satisfied: cheroot>=8.2.1 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (9.0.0)\n",
      "Requirement already satisfied: jaraco.collections in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (4.1.0)\n",
      "Requirement already satisfied: pywin32>=227 in d:\\anaconda\\setup\\lib\\site-packages (from cherrypy->pattern3) (302)\n",
      "Requirement already satisfied: Pillow>=2.0 in d:\\anaconda\\setup\\lib\\site-packages (from docx->pattern3) (9.2.0)\n",
      "Requirement already satisfied: lxml in d:\\anaconda\\setup\\lib\\site-packages (from docx->pattern3) (4.9.1)\n",
      "Requirement already satisfied: sgmllib3k in d:\\anaconda\\setup\\lib\\site-packages (from feedparser->pattern3) (1.0.0)\n",
      "Requirement already satisfied: charset-normalizer>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer.six->pattern3) (2.0.4)\n",
      "Requirement already satisfied: cryptography>=36.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer.six->pattern3) (38.0.1)\n",
      "Requirement already satisfied: ply in d:\\anaconda\\setup\\lib\\site-packages (from pdfminer3k->pattern3) (3.11)\n",
      "Requirement already satisfied: jaraco.functools in d:\\anaconda\\setup\\lib\\site-packages (from cheroot>=8.2.1->cherrypy->pattern3) (3.6.0)\n",
      "Requirement already satisfied: six>=1.11.0 in d:\\anaconda\\setup\\lib\\site-packages (from cheroot>=8.2.1->cherrypy->pattern3) (1.16.0)\n",
      "Requirement already satisfied: cffi>=1.12 in d:\\anaconda\\setup\\lib\\site-packages (from cryptography>=36.0.0->pdfminer.six->pattern3) (1.15.1)\n",
      "Requirement already satisfied: tempora>=1.8 in d:\\anaconda\\setup\\lib\\site-packages (from portend>=2.1.1->cherrypy->pattern3) (5.2.2)\n",
      "Requirement already satisfied: jaraco.text in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.collections->cherrypy->pattern3) (3.11.1)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda\\setup\\lib\\site-packages (from zc.lockfile->cherrypy->pattern3) (65.5.0)\n",
      "Requirement already satisfied: pycparser in d:\\anaconda\\setup\\lib\\site-packages (from cffi>=1.12->cryptography>=36.0.0->pdfminer.six->pattern3) (2.21)\n",
      "Requirement already satisfied: pytz in d:\\anaconda\\setup\\lib\\site-packages (from tempora>=1.8->portend>=2.1.1->cherrypy->pattern3) (2022.1)\n",
      "Requirement already satisfied: inflect in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (6.0.4)\n",
      "Requirement already satisfied: autocommand in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (2.2.2)\n",
      "Requirement already satisfied: jaraco.context>=4.1 in d:\\anaconda\\setup\\lib\\site-packages (from jaraco.text->jaraco.collections->cherrypy->pattern3) (4.3.0)\n",
      "Requirement already satisfied: pydantic>=1.9.1 in d:\\anaconda\\setup\\lib\\site-packages (from inflect->jaraco.text->jaraco.collections->cherrypy->pattern3) (1.10.7)\n",
      "Requirement already satisfied: typing-extensions>=4.2.0 in d:\\anaconda\\setup\\lib\\site-packages (from pydantic>=1.9.1->inflect->jaraco.text->jaraco.collections->cherrypy->pattern3) (4.3.0)\n",
      "Requirement already satisfied: pyLDAvis in d:\\anaconda\\setup\\lib\\site-packages (3.4.1)\n",
      "Requirement already satisfied: numpy>=1.24.2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.24.3)\n",
      "Requirement already satisfied: scikit-learn>=1.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.0.2)\n",
      "Requirement already satisfied: numexpr in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.8.3)\n",
      "Requirement already satisfied: joblib>=1.2.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.2.0)\n",
      "Requirement already satisfied: gensim in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (4.1.2)\n",
      "Requirement already satisfied: pandas>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (65.5.0)\n",
      "Requirement already satisfied: jinja2 in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.11.3)\n",
      "Requirement already satisfied: funcy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (2.0)\n",
      "Requirement already satisfied: scipy in d:\\anaconda\\setup\\lib\\site-packages (from pyLDAvis) (1.10.1)\n",
      "Requirement already satisfied: pytz>=2020.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2022.1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2023.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in d:\\anaconda\\setup\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2.8.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in d:\\anaconda\\setup\\lib\\site-packages (from scikit-learn>=1.0.0->pyLDAvis) (2.2.0)\n",
      "Requirement already satisfied: smart-open>=1.8.1 in d:\\anaconda\\setup\\lib\\site-packages (from gensim->pyLDAvis) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in d:\\anaconda\\setup\\lib\\site-packages (from jinja2->pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: packaging in d:\\anaconda\\setup\\lib\\site-packages (from numexpr->pyLDAvis) (21.3)\n",
      "Requirement already satisfied: six>=1.5 in d:\\anaconda\\setup\\lib\\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->pyLDAvis) (1.16.0)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in d:\\anaconda\\setup\\lib\\site-packages (from packaging->numexpr->pyLDAvis) (3.0.9)\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  ['<', 'p', '>', 'The', 'circus', 'dog', 'in', 'a', 'plissé', 'skirt', 'jumped', 'over', 'Python', 'who', 'was', \"n't\", 'that', 'large', ',', 'just', '3', 'feet', 'long.', '<', '/p', '>']\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  <p>The circus dog in a plissé skirt jumped over Python who was not that large, just 3 feet long.</p>\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  [('<', 'a'), ('p', 'n'), ('>', 'v'), ('the', None), ('circus', 'n'), ('dog', 'n'), ('in', None), ('a', None), ('plissé', 'n'), ('skirt', 'n'), ('jumped', 'v'), ('over', None), ('python', 'n'), ('who', None), ('was', 'v'), (\"n't\", 'r'), ('that', None), ('large', 'a'), (',', None), ('just', 'r'), ('3', None), ('feet', 'n'), ('long.', 'a'), ('<', 'n'), ('/p', 'n'), ('>', 'n')]\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  < p > the circus dog in a plissé skirt jump over python who be n't that large , just 3 foot long. < /p >\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:    p   The circus dog in a plissé skirt jumped over Python who was n t that large   just 3 feet long     p  \n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  < p > The circus dog plissé skirt jumped Python n't large , 3 feet long. < /p >\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  p The circus dog in a plissé skirt jumped over Python who was n't that large just feet long. /p\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.\n",
      "Original:   <p>The circus dog in a plissé skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n",
      "Processed:  <p>The circus dog in a plisse skirt jumped over Python who wasn't that large, just 3 feet long.</p>\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package averaged_perceptron_tagger is already up-to-\n",
      "[nltk_data]       date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\drrr8\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "#the module 'sys' allows istalling module from inside Jupyter\n",
    "import sys\n",
    "\n",
    "!{sys.executable} -m pip install numpy\n",
    "import numpy as np\n",
    "\n",
    "!{sys.executable} -m pip install pandas\n",
    "import pandas as pd\n",
    "\n",
    "#Natrual Language ToolKit (NLTK)\n",
    "!{sys.executable} -m pip install nltk\n",
    "import nltk\n",
    "\n",
    "!{sys.executable} -m pip install sklearn\n",
    "from sklearn import metrics\n",
    "#from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.feature_extraction.text import  CountVectorizer #bag-of-words vectorizer \n",
    "from sklearn.decomposition import LatentDirichletAllocation #package for LDA\n",
    "\n",
    "# Plotting tools\n",
    "\n",
    "from pprint import pprint\n",
    "!{sys.executable} -m pip install pyLDAvis #visualizing LDA\n",
    "import pyLDAvis\n",
    "import pyLDAvis.lda_model\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "#define text normalization function\n",
    "%run \"C:/Users/drrr8/OneDrive - Washington University in St. Louis/Desktop/562 text mining/Text_Normalization_Function.ipynb\"\n",
    "#ignore warnings about future changes in functions as they take too much space\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning)\n",
    "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "8381e6a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_topics(model, feature_names, no_top_words):\n",
    "    for topic_idx, topic in enumerate(model.components_):\n",
    "        print(\"Topic %d:\" % (topic_idx))\n",
    "        print(\" \".join([feature_names[i]\n",
    "                        for i in topic.argsort()[:-no_top_words - 1:-1]]))\n",
    "        \n",
    "def get_topic_words(vectorizer, lda_model, n_words):\n",
    "    keywords = np.array(vectorizer.get_feature_names_out())\n",
    "    topic_words = []\n",
    "    for topic_weights in lda_model.components_:\n",
    "        top_word_locs = (-topic_weights).argsort()[:n_words]\n",
    "        topic_words.append(keywords.take(top_word_locs).tolist())\n",
    "    return topic_words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "20846b1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#define the bag-of-words vectorizer:\n",
    "bow_vectorizer = CountVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "8dd718d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# do LDA for positive\n",
    "# vectorize the normalized data:\n",
    "bow_positive_corpus = bow_vectorizer.fit_transform(positive_corpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "29bc67e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# remember to change parameters\n",
    "lda_positive_corpus = LatentDirichletAllocation(n_components=5, max_iter=500,\n",
    "                                                doc_topic_prior = 0.9,\n",
    "                                                topic_word_prior = 0.1).fit(bow_positive_corpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "517881ce",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Topic 0:\n",
      "not room hotel no did night check day desk just got arrived stay asked time\n",
      "Topic 1:\n",
      "not beach resort food did good pool great time just people day nice like no\n",
      "Topic 2:\n",
      "hotel great location good breakfast staff clean stay walk rooms stayed helpful nice friendly room\n",
      "Topic 3:\n",
      "room bed bathroom nice not rooms view floor small shower new night tv stay stayed\n",
      "Topic 4:\n",
      "hotel staff stay service great wonderful stayed best beautiful experience fantastic rooms friendly not loved\n"
     ]
    }
   ],
   "source": [
    "no_top_words = 15\n",
    "display_topics(lda_positive_corpus, bow_vectorizer.get_feature_names_out(), no_top_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "ec7dcd3d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Topic_0</th>\n",
       "      <th>Topic_1</th>\n",
       "      <th>Topic_2</th>\n",
       "      <th>Topic_3</th>\n",
       "      <th>Topic_4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>00</th>\n",
       "      <td>1.845213e-03</td>\n",
       "      <td>8.242459e-04</td>\n",
       "      <td>7.783629e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>000</th>\n",
       "      <td>1.271189e-04</td>\n",
       "      <td>2.235354e-05</td>\n",
       "      <td>7.783543e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000170</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0001</th>\n",
       "      <td>8.959180e-06</td>\n",
       "      <td>5.620420e-07</td>\n",
       "      <td>7.782538e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>000rp</th>\n",
       "      <td>8.145304e-07</td>\n",
       "      <td>5.620995e-07</td>\n",
       "      <td>7.782027e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000036</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>000rupiah</th>\n",
       "      <td>8.145265e-07</td>\n",
       "      <td>5.621135e-07</td>\n",
       "      <td>7.782743e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>000us</th>\n",
       "      <td>8.145425e-07</td>\n",
       "      <td>5.620477e-07</td>\n",
       "      <td>7.786253e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>00a</th>\n",
       "      <td>8.146858e-07</td>\n",
       "      <td>2.304311e-05</td>\n",
       "      <td>7.783318e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>00am</th>\n",
       "      <td>1.586972e-04</td>\n",
       "      <td>7.146756e-05</td>\n",
       "      <td>7.784552e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>00dollars</th>\n",
       "      <td>8.145154e-07</td>\n",
       "      <td>5.625193e-07</td>\n",
       "      <td>7.781946e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>00gbp</th>\n",
       "      <td>8.958955e-06</td>\n",
       "      <td>5.620639e-07</td>\n",
       "      <td>7.784225e-07</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Topic_0       Topic_1       Topic_2   Topic_3   Topic_4\n",
       "00         1.845213e-03  8.242459e-04  7.783629e-07  0.000001  0.000001\n",
       "000        1.271189e-04  2.235354e-05  7.783543e-07  0.000001  0.000170\n",
       "0001       8.959180e-06  5.620420e-07  7.782538e-07  0.000001  0.000001\n",
       "000rp      8.145304e-07  5.620995e-07  7.782027e-07  0.000001  0.000036\n",
       "000rupiah  8.145265e-07  5.621135e-07  7.782743e-07  0.000001  0.000013\n",
       "000us      8.145425e-07  5.620477e-07  7.786253e-07  0.000001  0.000013\n",
       "00a        8.146858e-07  2.304311e-05  7.783318e-07  0.000001  0.000001\n",
       "00am       1.586972e-04  7.146756e-05  7.784552e-07  0.000001  0.000001\n",
       "00dollars  8.145154e-07  5.625193e-07  7.781946e-07  0.000001  0.000013\n",
       "00gbp      8.958955e-06  5.620639e-07  7.784225e-07  0.000001  0.000001"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "word_weights = lda_positive_corpus.components_ / lda_positive_corpus.components_.sum(axis=1)[:, np.newaxis]\n",
    "word_weights_df = pd.DataFrame(word_weights.T, \n",
    "                               index = bow_vectorizer.get_feature_names_out(), \n",
    "                               columns = [\"Topic_\" + str(i) for i in range(5)])\n",
    "word_weights_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "e77d6a8c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Topic_0</th>\n",
       "      <th>Topic_1</th>\n",
       "      <th>Topic_2</th>\n",
       "      <th>Topic_3</th>\n",
       "      <th>Topic_4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>not</th>\n",
       "      <td>0.022846</td>\n",
       "      <td>1.752572e-02</td>\n",
       "      <td>4.400524e-03</td>\n",
       "      <td>0.013675</td>\n",
       "      <td>0.005874</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>room</th>\n",
       "      <td>0.019088</td>\n",
       "      <td>2.657188e-03</td>\n",
       "      <td>8.849718e-03</td>\n",
       "      <td>0.060533</td>\n",
       "      <td>0.002463</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>hotel</th>\n",
       "      <td>0.017970</td>\n",
       "      <td>5.621860e-07</td>\n",
       "      <td>6.660486e-02</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>0.032563</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>no</th>\n",
       "      <td>0.011626</td>\n",
       "      <td>5.619091e-03</td>\n",
       "      <td>6.418654e-04</td>\n",
       "      <td>0.005193</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>did</th>\n",
       "      <td>0.010481</td>\n",
       "      <td>1.088965e-02</td>\n",
       "      <td>1.269123e-03</td>\n",
       "      <td>0.005338</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>night</th>\n",
       "      <td>0.008288</td>\n",
       "      <td>3.453785e-03</td>\n",
       "      <td>3.157760e-03</td>\n",
       "      <td>0.007678</td>\n",
       "      <td>0.000460</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>check</th>\n",
       "      <td>0.006307</td>\n",
       "      <td>9.825819e-04</td>\n",
       "      <td>7.783977e-07</td>\n",
       "      <td>0.001821</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>day</th>\n",
       "      <td>0.006082</td>\n",
       "      <td>6.811331e-03</td>\n",
       "      <td>1.793080e-03</td>\n",
       "      <td>0.001968</td>\n",
       "      <td>0.003388</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>desk</th>\n",
       "      <td>0.005909</td>\n",
       "      <td>5.622129e-07</td>\n",
       "      <td>7.784071e-07</td>\n",
       "      <td>0.005133</td>\n",
       "      <td>0.000001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>just</th>\n",
       "      <td>0.005400</td>\n",
       "      <td>7.851490e-03</td>\n",
       "      <td>6.017439e-03</td>\n",
       "      <td>0.002166</td>\n",
       "      <td>0.005429</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Topic_0       Topic_1       Topic_2   Topic_3   Topic_4\n",
       "not    0.022846  1.752572e-02  4.400524e-03  0.013675  0.005874\n",
       "room   0.019088  2.657188e-03  8.849718e-03  0.060533  0.002463\n",
       "hotel  0.017970  5.621860e-07  6.660486e-02  0.000001  0.032563\n",
       "no     0.011626  5.619091e-03  6.418654e-04  0.005193  0.000001\n",
       "did    0.010481  1.088965e-02  1.269123e-03  0.005338  0.000001\n",
       "night  0.008288  3.453785e-03  3.157760e-03  0.007678  0.000460\n",
       "check  0.006307  9.825819e-04  7.783977e-07  0.001821  0.000001\n",
       "day    0.006082  6.811331e-03  1.793080e-03  0.001968  0.003388\n",
       "desk   0.005909  5.622129e-07  7.784071e-07  0.005133  0.000001\n",
       "just   0.005400  7.851490e-03  6.017439e-03  0.002166  0.005429"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "word_weights_df.sort_values(by='Topic_0',ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "03e686c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<link rel=\"stylesheet\" type=\"text/css\" href=\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css\">\n",
       "\n",
       "\n",
       "<div id=\"ldavis_el1467229661875813287835259874\" style=\"background-color:white;\"></div>\n",
       "<script type=\"text/javascript\">\n",
       "\n",
       "var ldavis_el1467229661875813287835259874_data = {\"mdsDat\": {\"x\": [-83.1103286743164, 89.41065216064453, -126.83918762207031, 49.30977249145508, 201.81375122070312], \"y\": [-142.0851287841797, -180.25900268554688, 35.64542770385742, 55.00224304199219, -35.33937072753906], \"topics\": [1, 2, 3, 4, 5], \"cluster\": [1, 1, 1, 1, 1], \"Freq\": [29.862519947757253, 21.05990876760911, 20.56423258904457, 14.424794327444685, 14.08854436814439]}, \"tinfo\": {\"Term\": [\"hotel\", \"room\", \"location\", \"beach\", \"great\", \"staff\", \"resort\", \"bed\", \"bathroom\", \"nice\", \"clean\", \"stay\", \"good\", \"walk\", \"service\", \"food\", \"helpful\", \"shower\", \"stayed\", \"breakfast\", \"small\", \"rooms\", \"floor\", \"wonderful\", \"city\", \"friendly\", \"beautiful\", \"view\", \"pool\", \"no\", \"beach\", \"resort\", \"kids\", \"ocean\", \"cana\", \"punta\", \"entertainment\", \"grounds\", \"pools\", \"dominican\", \"resorts\", \"activities\", \"shows\", \"swim\", \"sand\", \"casino\", \"country\", \"inclusive\", \"sick\", \"bavaro\", \"mexican\", \"boat\", \"steak\", \"beaches\", \"tropical\", \"fish\", \"golf\", \"seafood\", \"riu\", \"pizza\", \"fun\", \"vacation\", \"spanish\", \"food\", \"pool\", \"people\", \"drinks\", \"lunch\", \"water\", \"buffet\", \"awesome\", \"did\", \"went\", \"time\", \"drink\", \"not\", \"bring\", \"good\", \"day\", \"beautiful\", \"just\", \"like\", \"restaurants\", \"great\", \"got\", \"really\", \"bar\", \"nice\", \"no\", \"little\", \"want\", \"best\", \"service\", \"staff\", \"place\", \"station\", \"distance\", \"metro\", \"central\", \"barcelona\", \"florence\", \"blocks\", \"amsterdam\", \"centre\", \"downtown\", \"ramblas\", \"attractions\", \"subway\", \"district\", \"boston\", \"canal\", \"sights\", \"museum\", \"neighborhood\", \"cafes\", \"berlin\", \"sightseeing\", \"bridge\", \"furnished\", \"bonus\", \"museums\", \"catalunya\", \"duomo\", \"madrid\", \"stylish\", \"location\", \"train\", \"value\", \"city\", \"ideal\", \"walking\", \"modern\", \"located\", \"walk\", \"helpful\", \"hotel\", \"shopping\", \"breakfast\", \"clean\", \"easy\", \"close\", \"street\", \"great\", \"excellent\", \"convenient\", \"good\", \"friendly\", \"nights\", \"staff\", \"stay\", \"stayed\", \"comfortable\", \"rooms\", \"nice\", \"recommend\", \"price\", \"small\", \"room\", \"the\", \"area\", \"just\", \"restaurants\", \"credit\", \"saturday\", \"arriving\", \"it__\\u00e7_\\u00e9_\", \"didn__\\u00e7_\\u00e9_\", \"carpet\", \"cereal\", \"iron\", \"receptionist\", \"calls\", \"clearly\", \"checkout\", \"croissants\", \"fare\", \"port\", \"clerk\", \"supermarket\", \"fix\", \"ticket\", \"bellman\", \"curtain\", \"computers\", \"omni\", \"informed\", \"mistake\", \"suggested\", \"unable\", \"shampoo\", \"toronto\", \"priceline\", \"card\", \"bags\", \"phone\", \"charge\", \"asked\", \"told\", \"checked\", \"charged\", \"check\", \"friday\", \"called\", \"arrived\", \"later\", \"desk\", \"given\", \"reservation\", \"no\", \"ready\", \"said\", \"not\", \"morning\", \"finally\", \"flight\", \"night\", \"left\", \"did\", \"hours\", \"room\", \"decided\", \"came\", \"car\", \"got\", \"hotel\", \"day\", \"way\", \"minutes\", \"just\", \"booked\", \"like\", \"time\", \"took\", \"stay\", \"people\", \"service\", \"make\", \"floor\", \"good\", \"san\", \"juan\", \"bali\", \"francisco\", \"ritz\", \"ubud\", \"delightful\", \"library\", \"magnificent\", \"memorable\", \"top\", \"kuta\", \"argonaut\", \"oriental\", \"chic\", \"celebrate\", \"owners\", \"mandarin\", \"recommendation\", \"dua\", \"nusa\", \"arc\", \"castle\", \"condado\", \"shangri\", \"raffles\", \"impeccable\", \"david\", \"balinese\", \"bravo\", \"exceptional\", \"class\", \"outstanding\", \"fabulous\", \"wonderful\", \"thanks\", \"absolutely\", \"special\", \"superb\", \"pleasure\", \"birthday\", \"attentive\", \"moment\", \"experience\", \"welcome\", \"fantastic\", \"truly\", \"service\", \"staff\", \"love\", \"years\", \"best\", \"amazing\", \"beautiful\", \"loved\", \"return\", \"stay\", \"hotel\", \"perfect\", \"feel\", \"year\", \"visit\", \"lovely\", \"stayed\", \"staying\", \"family\", \"trip\", \"great\", \"friendly\", \"home\", \"hotels\", \"old\", \"rooms\", \"excellent\", \"just\", \"time\", \"not\", \"place\", \"the\", \"say\", \"like\", \"tv\", \"bath\", \"bedroom\", \"pillows\", \"nyc\", \"flat\", \"screen\", \"orleans\", \"waikiki\", \"westin\", \"quarter\", \"kitchen\", \"closet\", \"sink\", \"queen\", \"channels\", \"twin\", \"elevators\", \"sofa\", \"microwave\", \"mattress\", \"furnishings\", \"robes\", \"designed\", \"unit\", \"cramped\", \"bourbon\", \"upstairs\", \"plasma\", \"motel\", \"shower\", \"bed\", \"bathroom\", \"window\", \"separate\", \"space\", \"king\", \"tub\", \"suite\", \"marble\", \"wall\", \"room\", \"floor\", \"view\", \"new\", \"small\", \"double\", \"large\", \"size\", \"nice\", \"beds\", \"lobby\", \"rooms\", \"noise\", \"door\", \"not\", \"night\", \"desk\", \"big\", \"the\", \"stayed\", \"area\", \"stay\", \"comfortable\", \"no\", \"really\", \"did\", \"service\", \"place\", \"like\"], \"Freq\": [13146.0, 9068.0, 3038.0, 2854.0, 5961.0, 4521.0, 2210.0, 1303.0, 1389.0, 3509.0, 2608.0, 4046.0, 4759.0, 1707.0, 2733.0, 2498.0, 1588.0, 838.0, 2838.0, 2671.0, 1771.0, 3336.0, 1285.0, 1165.0, 1176.0, 1953.0, 1371.0, 1365.0, 2167.0, 2894.0, 2854.158829451406, 2210.519008595237, 611.2609166998092, 544.338147109124, 450.8431145846049, 441.0015271317955, 359.3162808055331, 351.4430458475211, 301.25093555373377, 291.40939295027533, 287.47273274585086, 276.646980190759, 230.39151105036413, 213.66080444224528, 208.7400415728971, 204.80334234040018, 196.9300759263162, 192.009315871753, 186.10438943796308, 159.53211547895984, 153.62715135598896, 153.62707948024666, 150.6746240518718, 148.7063277375747, 147.7221638724213, 146.73798729977295, 144.76971378173116, 144.76966725015504, 140.83309408048004, 136.89637780244493, 619.4756292474162, 603.9718862942262, 295.04518279430874, 2072.636395255691, 1782.1279896198298, 1287.1590610314595, 580.6531455136369, 353.7277100818219, 968.911858106326, 733.6803121327605, 263.45015332890154, 1906.8277729274705, 789.8754241804162, 1401.4718835423153, 389.56068969295757, 3068.832666511171, 332.0761882869264, 1872.2759067207185, 1192.6952017496005, 755.7522740388539, 1374.8317676774845, 1030.6969023213187, 727.2007082752388, 1643.7780298190419, 773.6256098347884, 877.3464782023373, 720.4769827253091, 1091.4886729195136, 983.9285332589264, 693.6677229686272, 588.3523015769856, 631.6362878522043, 690.8188364169207, 729.973653059311, 651.2667194800373, 714.0988434978526, 486.3482462473356, 427.7289608228022, 403.70461310894825, 380.64124843347406, 301.8415084707404, 290.30980531157854, 288.38789093504556, 280.70006936705937, 224.00268020749306, 193.25158118249527, 171.14919207926582, 170.18822738041132, 150.96875504156813, 147.1248350113339, 143.28097785096818, 141.3589985604701, 125.98347150859513, 125.02248437700382, 122.1395950645615, 114.45181023230336, 111.56886725921008, 111.56886034126849, 98.11525591291665, 95.23226776328734, 92.34943677918719, 90.42748365769681, 87.54457392734268, 85.62261847486634, 81.77869329515728, 2951.0475536898043, 331.2668112184991, 617.7074876110609, 1051.6118761715074, 166.82734337231, 697.9219514851596, 521.0963921067055, 579.6042384031228, 1364.2347741080068, 1267.9085414835265, 8224.952245349681, 458.07961240474987, 1870.3416621100118, 1789.572916642422, 501.7728412577505, 654.5335384810082, 760.9503608736023, 3162.772267144877, 1085.908492546114, 296.9616407986242, 2341.6843796356843, 1129.3039681594958, 753.1687937151927, 1803.3654934054937, 1663.0829456765666, 1277.063060164413, 738.3612925608292, 1331.0487768368773, 1253.1640619680843, 718.7608669203627, 625.2364967519018, 697.7688037395104, 1092.8407998850773, 710.5131721315212, 648.9583431181201, 743.0861916144519, 617.9750513800849, 115.98988271582571, 111.07917461850569, 99.29356856319326, 93.4007881302783, 87.50800737890972, 81.61518326715503, 76.7044956661241, 75.72238083423711, 73.75814420906076, 71.79385501580282, 69.82957871794747, 69.82957776794298, 66.88316827036849, 63.93677069951522, 62.95462994658709, 60.99040721482263, 60.00824001244289, 54.11546562993663, 54.11543752555202, 51.16904638650305, 51.169032939418884, 51.169031681395936, 50.18693701576687, 50.18693015948697, 48.22265714332802, 48.22262930532878, 47.24052501224921, 47.240510687193186, 47.24050592040187, 44.2941542046296, 207.90880868381944, 230.80653064634828, 197.44006055153963, 250.09231481586022, 500.9266109598425, 495.2924169571247, 293.3623686845938, 95.57693662632809, 760.4822945725358, 90.75584641917, 299.18888710769954, 528.9296646041586, 230.35884754951715, 712.5574918219238, 273.6106097139775, 185.90853868884187, 1401.8521459373765, 243.87473399086988, 391.42130706783416, 2754.77096932709, 476.03886074909093, 156.40443224114998, 194.2444482526819, 999.3267972676822, 372.1300508690571, 1263.8261106996472, 289.51230648443067, 2301.72323355619, 191.45506453803716, 276.5855863883852, 273.03284731030175, 587.8037107720337, 2166.8990067869304, 733.3752391305756, 444.68428543937995, 388.46478331185943, 651.1325837634613, 362.5791362071817, 495.0401730646069, 499.7970411776525, 317.70073556807364, 501.91697259021106, 379.94095878400645, 387.2195329178816, 319.8065838860275, 312.66703851435733, 298.3015753769813, 436.7216011652882, 237.90164650332304, 184.29825649725723, 159.93302999906683, 117.05027764392449, 79.04063858892262, 64.4214346472114, 61.49765993857111, 60.523049422219174, 53.70083079940697, 49.802311336459546, 48.827765838357834, 46.87857909758051, 46.878533964775414, 46.87851499674533, 45.90393974253298, 45.9039183689223, 42.00551223269358, 41.03089043245545, 38.107126658634456, 38.10712561286807, 38.10704522989806, 37.13247840329204, 36.157866667436224, 34.208673716173564, 34.2086653414419, 34.20865986481451, 34.20860856499898, 31.284858653989858, 30.310243075518407, 101.2127074625852, 129.1074062133277, 164.98165212345006, 297.09771991991, 814.816031535906, 131.49333825230812, 308.8245072527598, 302.80281652762005, 184.22137319167433, 77.15542166992563, 121.00080749068927, 145.96902164106527, 111.58186646991871, 570.2204511085274, 149.139320939515, 538.5817269176667, 151.04198781922284, 1241.7970349053421, 1712.294596734636, 200.45985462523578, 213.1541267333764, 717.2428920148882, 352.3437689221324, 615.2531588156652, 480.2724472238109, 343.97074776468014, 1299.6229611262781, 2754.23801544218, 404.4683236293125, 333.6698431673658, 302.17239825698516, 298.61817064861503, 407.45728757643684, 802.2582618298836, 321.5201945933936, 300.0025611757923, 469.0812129359672, 1023.7016592884029, 499.83536714575934, 282.00287166607154, 352.13468961894824, 303.81502598152446, 509.2294463336763, 396.47757683373675, 459.1757483123425, 415.8466619399945, 496.85404953899433, 389.3705336375989, 376.0765356960292, 336.0710371995266, 362.7654592210302, 551.1291229750885, 328.1629144137902, 262.3535266372306, 196.544098480955, 177.88173751558298, 174.93505779472434, 171.00614212419745, 168.05945141933347, 141.53925553530254, 141.53923775911525, 138.59256455158655, 131.71694107787837, 116.00126792169351, 115.0190257052767, 113.05457046780081, 105.19673795936325, 101.26779807811538, 96.35666860770385, 74.74762064659936, 67.87197415009537, 65.90750346351837, 63.94308903557506, 60.996369080147886, 58.049679172902124, 55.103008859845694, 54.12078375862109, 53.13857048267744, 51.17406671306036, 49.20963407026785, 48.227377439325714, 816.7328917623686, 1255.8952821247026, 1192.4229569676625, 228.82715462960442, 155.98001810302728, 274.86845008522914, 259.62821452978983, 214.18760756785557, 455.3331966656153, 105.38694322228193, 186.4632533945866, 5000.651545935027, 843.7361630044495, 853.2559143277178, 645.3851249164101, 843.1080152065304, 257.82318328232424, 515.7790007186317, 316.79699800152235, 1165.10617014673, 391.76995413513, 440.31507534082215, 909.0376654860544, 319.64651681564703, 331.13723919103415, 1129.6652529909866, 634.326341539678, 424.0089253065789, 352.6376134748971, 491.20308436414405, 521.385384206206, 423.19502357274894, 527.8351576787551, 372.38840207997697, 429.0357932935853, 400.9662907375383, 440.99684634764924, 413.56518403155496, 375.6049798753493, 368.18902551313136], \"Total\": [13146.0, 9068.0, 3038.0, 2854.0, 5961.0, 4521.0, 2210.0, 1303.0, 1389.0, 3509.0, 2608.0, 4046.0, 4759.0, 1707.0, 2733.0, 2498.0, 1588.0, 838.0, 2838.0, 2671.0, 1771.0, 3336.0, 1285.0, 1165.0, 1176.0, 1953.0, 1371.0, 1365.0, 2167.0, 2894.0, 2854.5488870613717, 2210.9090417449283, 611.6510149339163, 544.7282179404516, 451.2331337670698, 441.3915459173708, 359.7063660964277, 351.83309614420466, 301.6409979740603, 291.79941057475133, 287.86277518110853, 277.0370285131545, 230.78156550416622, 214.05086609238734, 209.13007248909136, 205.19343675367026, 197.3201666096922, 192.39937284440828, 186.49442046003912, 159.9221333398383, 154.01718051432476, 154.0171796339801, 151.0647036779297, 149.09638630673567, 148.1122275205702, 147.1280684746001, 145.1597513498788, 145.15975079727644, 141.22311636089856, 137.28648032784432, 638.0451443630749, 649.5899745458315, 308.5031612463791, 2498.2934727843567, 2167.292274273907, 1766.4739743324499, 710.4581305039287, 410.53525594353874, 1356.8754455814121, 978.7438216558281, 294.02363464166183, 3768.470694317948, 1230.5466617366997, 2669.8771171430503, 488.13751244109835, 7993.53830541262, 396.9727932402475, 4759.364887225588, 2596.562645523192, 1371.2980406845368, 3407.146751495133, 2256.787684274662, 1421.7052176675284, 5961.547740499024, 1600.8765760745237, 2171.963816457577, 1538.3750097603172, 3509.954634185475, 2894.177137634207, 1734.6059885293016, 1156.199917735472, 1652.326834710403, 2733.496714556291, 4521.874488629098, 2136.7349461376743, 714.4912416816285, 486.74067617517284, 428.12133090056795, 404.0970101228806, 381.03366200349507, 302.2338872584347, 290.7022137482448, 288.7802673338101, 281.0924853694244, 224.39508688897092, 193.64395488783973, 171.54157957699553, 170.58060655862712, 151.36114978621845, 147.51725899443719, 143.6733668097886, 141.7514217782715, 126.3758553790173, 125.41488281603363, 122.53196367504844, 114.84418088661003, 111.96126274065323, 111.96126294443421, 98.50764269991285, 95.6247257352618, 92.74180519449665, 90.81985970489674, 87.9369409120898, 86.01499545368891, 82.1711046520089, 3038.076615590607, 344.7998152104369, 655.7990852561106, 1176.5394912270149, 172.57750810242456, 796.566729701956, 587.1417179632746, 667.364006022201, 1707.5678012110143, 1588.2248804585397, 13146.285961107018, 533.6781922988908, 2671.8729852599263, 2608.1542193933965, 611.3566840104689, 847.6425031239585, 1029.7329100391307, 5961.547740499024, 1728.1134796822498, 343.59548958376956, 4759.364887225588, 1953.3942284202917, 1161.5731292844955, 4521.874488629098, 4046.5477729899944, 2838.4737730943016, 1239.4257380426418, 3336.2818436707144, 3509.954634185475, 1314.6785785485445, 1016.1635599840552, 1771.1319710728335, 9068.85817997771, 2187.0646011322306, 1644.03998547958, 3407.146751495133, 1421.7052176675284, 116.38016037126332, 111.46949262445325, 99.68389093619412, 93.79109019647306, 87.89828957377856, 82.0054884161974, 77.09482069340754, 76.11268751130073, 74.14842061452022, 72.18415366906929, 70.21988678679618, 70.21988661084869, 67.27348579861365, 64.32708571017406, 63.344951930297725, 61.38068548068265, 60.398551494003165, 54.50575127587806, 54.505750726115245, 51.55935052132024, 51.55935040729288, 51.559350233144414, 50.57721696248874, 50.577217143435774, 48.61295029872404, 48.61294998123368, 47.63081674152253, 47.63081663996023, 47.630816658018276, 44.684416212137386, 218.54312366881302, 245.06819201719037, 209.68400516004874, 277.4526997019584, 586.9984216625522, 586.0258817537884, 344.05532778500384, 103.62819038725488, 1083.1681387740423, 98.71733831730893, 385.6618838655127, 760.9367972558742, 290.3404195799519, 1136.8584781538045, 368.36778606210873, 238.27166698578995, 2894.177137634207, 334.5013865251987, 608.7421336533223, 7993.53830541262, 856.0460339668899, 203.88695139961823, 276.0990811695096, 2667.260468445491, 683.2151406013204, 3768.470694317948, 487.05019314189025, 9068.85817997771, 273.20202848142424, 470.70662679341154, 479.17369182633433, 1600.8765760745237, 13146.285961107018, 2596.562645523192, 1136.158514130805, 964.8528922674836, 3407.146751495133, 985.4730279025375, 2256.787684274662, 2669.8771171430503, 780.5412256036631, 4046.5477729899944, 1766.4739743324499, 2733.496714556291, 1098.5362842789245, 1285.3160252727598, 4759.364887225588, 437.1126403473549, 238.29267873018517, 184.6892574882702, 160.32406589734012, 117.44132934742629, 79.43163056850958, 64.81251611195036, 61.88869262405413, 60.91408537972647, 54.09183145458099, 50.193401215739975, 49.21879334386745, 47.26957772007478, 47.269578150170936, 47.26957818416774, 46.29497031477542, 46.294970361176816, 42.39653971571557, 41.42193213818492, 38.49810897178242, 38.49810897483293, 38.49810860222375, 37.52350119232506, 36.54889380390261, 34.59967835275931, 34.599678135971025, 34.59967840467793, 34.59967840967246, 31.6758555153, 30.70124789504162, 106.77079042771689, 140.94298047204813, 185.86271794927112, 352.4775787737154, 1165.9260021585278, 153.7136627777713, 399.9838358257508, 397.9849860316892, 228.8929336418775, 85.35505791214646, 142.9484113178641, 178.11341584683876, 131.23342461298952, 888.5851763392416, 188.88928479123416, 904.6930540221248, 198.78104978021972, 2733.496714556291, 4521.874488629098, 301.6457692286549, 328.9906954328509, 1652.326834710403, 640.6762505293034, 1371.2980406845368, 983.375848155735, 635.692254338458, 4046.5477729899944, 13146.285961107018, 838.4444845029121, 642.4941377514614, 561.8307047026638, 555.1800771622527, 904.0930295646365, 2838.4737730943016, 659.065255724814, 608.331785829408, 1336.5427735067167, 5961.547740499024, 1953.3942284202917, 591.9289382798918, 1049.2108845682055, 726.1882201358948, 3336.2818436707144, 1728.1134796822498, 3407.146751495133, 2669.8771171430503, 7993.53830541262, 2136.7349461376743, 2187.0646011322306, 989.843622129881, 2256.787684274662, 551.5194085890109, 328.5532109777137, 262.7438047184998, 196.93439845440082, 178.27202942550394, 175.32533978655928, 171.39641998037288, 168.4497301616229, 141.92952179903838, 141.92952163462925, 138.9828317219731, 132.10722215312552, 116.39154330270162, 115.40931328030945, 113.44485317304353, 105.58701384623332, 101.65809378300082, 96.74694426857012, 75.1378855781411, 68.2622757616728, 66.29781584841415, 64.33335584759023, 61.38666594316105, 58.43997614005257, 55.49328664924638, 54.51105645178751, 53.528826695447016, 51.56436683497138, 49.599906780812645, 48.61767671975658, 838.3284696202466, 1303.9028691364433, 1389.912782744172, 246.64267243911948, 165.52097464767502, 305.29745458794997, 288.33728504172706, 238.2344205211904, 559.5474831857288, 111.49159281336594, 211.6654105620792, 9068.85817997771, 1285.3160252727598, 1365.4443594931863, 1019.0116659092537, 1771.1319710728335, 354.7354886642644, 1034.1585172847217, 507.486759274968, 3509.954634185475, 704.4543134769559, 927.1891204271055, 3336.2818436707144, 577.5078721304728, 647.3204429390574, 7993.53830541262, 2667.260468445491, 1136.8584781538045, 856.2341288818209, 2187.0646011322306, 2838.4737730943016, 1644.03998547958, 4046.5477729899944, 1239.4257380426418, 2894.177137634207, 2171.963816457577, 3768.470694317948, 2733.496714556291, 2136.7349461376743, 2256.787684274662], \"Category\": [\"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Default\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic1\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic2\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic3\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic4\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\", \"Topic5\"], \"logprob\": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -4.1166, -4.3722, -5.6576, -5.7736, -5.962, -5.9841, -6.1889, -6.2111, -6.3652, -6.3984, -6.412, -6.4504, -6.6334, -6.7087, -6.732, -6.7511, -6.7903, -6.8156, -6.8468, -7.0009, -7.0386, -7.0386, -7.058, -7.0712, -7.0778, -7.0845, -7.098, -7.098, -7.1256, -7.1539, -5.6443, -5.6696, -6.386, -4.4366, -4.5876, -4.9129, -5.709, -6.2046, -5.197, -5.4751, -6.4993, -4.5199, -5.4013, -4.8279, -6.1081, -4.0441, -6.2678, -4.5382, -4.9892, -5.4454, -4.8471, -5.1351, -5.4839, -4.6684, -5.4221, -5.2962, -5.4932, -5.0778, -5.1816, -5.5311, -5.6958, -5.6248, -5.5353, -5.4801, -5.5942, -5.1529, -5.537, -5.6654, -5.7232, -5.782, -6.014, -6.053, -6.0596, -6.0866, -6.3122, -6.4599, -6.5814, -6.587, -6.7068, -6.7326, -6.7591, -6.7726, -6.8878, -6.8954, -6.9187, -6.9838, -7.0093, -7.0093, -7.1378, -7.1676, -7.1983, -7.2194, -7.2518, -7.274, -7.3199, -3.734, -5.921, -5.2979, -4.7658, -6.6069, -5.1758, -5.468, -5.3616, -4.5056, -4.5788, -2.709, -5.5969, -4.19, -4.2342, -5.5058, -5.24, -5.0893, -3.6647, -4.7337, -6.0303, -3.9653, -4.6945, -5.0996, -4.2265, -4.3075, -4.5716, -5.1195, -4.5302, -4.5905, -5.1464, -5.2858, -5.176, -4.7274, -5.1579, -5.2485, -5.1131, -5.2975, -6.9466, -6.9898, -7.102, -7.1632, -7.2284, -7.2981, -7.3601, -7.373, -7.3993, -7.4263, -7.454, -7.454, -7.4971, -7.5422, -7.5577, -7.5894, -7.6056, -7.709, -7.709, -7.765, -7.765, -7.765, -7.7843, -7.7843, -7.8243, -7.8243, -7.8448, -7.8448, -7.8448, -7.9092, -6.363, -6.2585, -6.4147, -6.1783, -5.4836, -5.4949, -6.0187, -7.1402, -5.0661, -7.1919, -5.999, -5.4292, -6.2604, -5.1312, -6.0884, -6.4748, -4.4545, -6.2034, -5.7303, -3.779, -5.5346, -6.6476, -6.431, -4.793, -5.7808, -4.5582, -6.0319, -3.9587, -6.4454, -6.0776, -6.0905, -5.3237, -4.019, -5.1024, -5.6027, -5.7379, -5.2214, -5.8068, -5.4954, -5.4859, -5.939, -5.4817, -5.7601, -5.7411, -5.9324, -5.9549, -6.002, -5.2662, -5.8736, -6.1289, -6.2707, -6.5829, -6.9755, -7.18, -7.2265, -7.2425, -7.3621, -7.4374, -7.4572, -7.4979, -7.4979, -7.4979, -7.5189, -7.5189, -7.6077, -7.6312, -7.7051, -7.7051, -7.7051, -7.731, -7.7576, -7.813, -7.813, -7.813, -7.813, -7.9023, -7.934, -6.7283, -6.4848, -6.2396, -5.6514, -4.6425, -6.4665, -5.6127, -5.6324, -6.1293, -6.9997, -6.5497, -6.3621, -6.6307, -4.9995, -6.3406, -5.0565, -6.3279, -4.2212, -3.8999, -6.0449, -5.9835, -4.7701, -5.4809, -4.9234, -5.1711, -5.5049, -4.1757, -3.4246, -5.3429, -5.5353, -5.6345, -5.6463, -5.3355, -4.6581, -5.5724, -5.6417, -5.1947, -4.4143, -5.1312, -5.7036, -5.4815, -5.6291, -5.1126, -5.3629, -5.216, -5.3152, -5.1372, -5.381, -5.4157, -5.5282, -5.4517, -5.0099, -5.5284, -5.7522, -6.041, -6.1408, -6.1575, -6.1802, -6.1976, -6.3693, -6.3693, -6.3904, -6.4412, -6.5683, -6.5768, -6.594, -6.6661, -6.7041, -6.7538, -7.0078, -7.1043, -7.1336, -7.1639, -7.2111, -7.2606, -7.3127, -7.3307, -7.349, -7.3867, -7.4258, -7.446, -4.6166, -4.1863, -4.2382, -5.8889, -6.2722, -5.7056, -5.7626, -5.955, -5.2009, -6.6643, -6.0937, -2.8046, -4.5841, -4.5728, -4.852, -4.5848, -5.7696, -5.0762, -5.5636, -4.2613, -5.3512, -5.2344, -4.5095, -5.5547, -5.5194, -4.2922, -4.8693, -5.2721, -5.4565, -5.125, -5.0654, -5.2741, -5.0531, -5.402, -5.2604, -5.328, -5.2329, -5.2971, -5.3934, -5.4133], \"loglift\": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.2084, 1.2084, 1.2079, 1.2078, 1.2077, 1.2077, 1.2075, 1.2075, 1.2073, 1.2072, 1.2072, 1.2072, 1.2069, 1.2067, 1.2067, 1.2067, 1.2066, 1.2065, 1.2065, 1.2061, 1.206, 1.206, 1.206, 1.2059, 1.2059, 1.2059, 1.2059, 1.2059, 1.2058, 1.2057, 1.179, 1.1358, 1.164, 1.0218, 1.0129, 0.892, 1.0068, 1.0596, 0.8718, 0.9204, 1.0988, 0.5273, 0.7652, 0.5641, 0.983, 0.2512, 1.0301, 0.2756, 0.4306, 0.6128, 0.301, 0.4249, 0.5382, -0.0798, 0.4813, 0.3021, 0.45, 0.0405, 0.1297, 0.292, 0.533, 0.2469, -0.1669, -0.6151, 0.0205, 1.5572, 1.557, 1.5569, 1.5568, 1.5568, 1.5565, 1.5564, 1.5564, 1.5564, 1.556, 1.5558, 1.5555, 1.5555, 1.5552, 1.5551, 1.5551, 1.555, 1.5547, 1.5547, 1.5546, 1.5544, 1.5543, 1.5543, 1.5538, 1.5537, 1.5536, 1.5535, 1.5533, 1.5532, 1.553, 1.5287, 1.5178, 1.498, 1.4455, 1.5239, 1.4256, 1.4385, 1.4168, 1.3333, 1.3326, 1.0888, 1.405, 1.2011, 1.1811, 1.3603, 1.2993, 1.2553, 0.9239, 1.0932, 1.4119, 0.8486, 1.0098, 1.1246, 0.6385, 0.6686, 0.7591, 1.0398, 0.6389, 0.5279, 0.954, 1.0721, 0.6263, -0.5583, 0.4335, 0.6283, 0.035, 0.7246, 1.5783, 1.5781, 1.5777, 1.5774, 1.5772, 1.5768, 1.5765, 1.5765, 1.5763, 1.5762, 1.576, 1.576, 1.5758, 1.5755, 1.5754, 1.5752, 1.5751, 1.5744, 1.5744, 1.574, 1.574, 1.574, 1.5739, 1.5739, 1.5736, 1.5736, 1.5734, 1.5734, 1.5734, 1.5728, 1.5317, 1.5217, 1.5215, 1.4778, 1.4231, 1.4134, 1.4222, 1.5007, 1.2279, 1.4975, 1.3277, 1.2179, 1.3502, 1.1145, 1.2842, 1.3335, 0.8567, 1.2656, 1.14, 0.5163, 0.9948, 1.3165, 1.23, 0.5999, 0.9741, 0.4891, 1.0614, 0.2104, 1.2261, 1.0499, 1.0191, 0.5797, -0.2212, 0.3173, 0.6436, 0.6718, -0.0733, 0.5817, 0.0646, -0.094, 0.6827, -0.5056, 0.0449, -0.3727, 0.3476, 0.168, -1.1881, 1.9353, 1.9346, 1.9341, 1.9338, 1.9329, 1.9313, 1.9302, 1.9299, 1.9298, 1.929, 1.9284, 1.9282, 1.9279, 1.9279, 1.9279, 1.9277, 1.9277, 1.927, 1.9267, 1.926, 1.926, 1.926, 1.9257, 1.9255, 1.9249, 1.9249, 1.9249, 1.9249, 1.9238, 1.9234, 1.8828, 1.8485, 1.817, 1.7653, 1.5779, 1.7801, 1.6776, 1.6629, 1.7191, 1.8352, 1.7695, 1.7372, 1.774, 1.4926, 1.6999, 1.4176, 1.6616, 1.1472, 0.9651, 1.5276, 1.5022, 1.1017, 1.3383, 1.1347, 1.2196, 1.3221, 0.8004, 0.3732, 1.2072, 1.281, 1.316, 1.3161, 1.1392, 0.6726, 1.2185, 1.2293, 0.8892, 0.1743, 0.5732, 1.1948, 0.8444, 1.0648, 0.0565, 0.4641, -0.068, 0.0768, -0.8419, 0.2337, 0.1757, 0.856, 0.1083, 1.9591, 1.9586, 1.9583, 1.9578, 1.9576, 1.9576, 1.9575, 1.9575, 1.9571, 1.9571, 1.957, 1.9568, 1.9564, 1.9564, 1.9564, 1.9561, 1.956, 1.9558, 1.9546, 1.9541, 1.9539, 1.9537, 1.9534, 1.9531, 1.9528, 1.9526, 1.9525, 1.9522, 1.9519, 1.9517, 1.9337, 1.9223, 1.8066, 1.8848, 1.9004, 1.8548, 1.8549, 1.8534, 1.7537, 1.9035, 1.833, 1.3645, 1.5389, 1.4896, 1.5031, 1.2175, 1.6407, 1.2641, 1.4886, 0.857, 1.3731, 1.2151, 0.6596, 1.3683, 1.2895, 0.0031, 0.5236, 0.9735, 1.0727, 0.4664, 0.2653, 0.6027, -0.077, 0.7573, 0.0509, 0.2703, -0.1856, 0.0713, 0.2213, 0.1467]}, \"token.table\": {\"Topic\": [1, 4, 1, 1, 4, 2, 4, 1, 2, 3, 4, 5, 4, 1, 3, 4, 3, 1, 3, 4, 5, 2, 1, 2, 1, 3, 4, 4, 1, 2, 3, 4, 5, 2, 5, 2, 5, 1, 1, 1, 1, 4, 3, 5, 5, 1, 2, 3, 5, 3, 2, 1, 2, 4, 1, 2, 5, 4, 5, 2, 1, 2, 1, 2, 3, 4, 5, 2, 5, 4, 1, 2, 3, 4, 5, 2, 1, 3, 5, 1, 2, 2, 1, 3, 3, 1, 3, 4, 1, 2, 2, 3, 4, 1, 3, 3, 1, 4, 2, 4, 2, 2, 3, 5, 3, 5, 1, 3, 1, 3, 5, 1, 3, 4, 3, 4, 2, 4, 1, 4, 1, 2, 5, 3, 3, 1, 2, 3, 5, 5, 2, 4, 5, 3, 4, 2, 5, 1, 5, 3, 3, 3, 4, 1, 2, 3, 4, 5, 1, 3, 4, 4, 5, 3, 5, 1, 2, 3, 5, 3, 2, 2, 1, 1, 2, 3, 5, 2, 3, 5, 2, 1, 3, 4, 1, 4, 5, 4, 2, 1, 2, 5, 1, 1, 2, 4, 1, 4, 1, 2, 3, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 3, 1, 3, 4, 5, 1, 3, 1, 3, 5, 1, 2, 3, 4, 2, 3, 5, 2, 1, 2, 4, 4, 1, 3, 1, 2, 4, 1, 4, 2, 5, 3, 4, 5, 1, 1, 2, 3, 5, 1, 2, 3, 5, 1, 2, 4, 5, 1, 1, 2, 4, 1, 3, 4, 2, 3, 4, 2, 3, 4, 5, 1, 3, 2, 4, 4, 1, 3, 3, 3, 4, 1, 2, 3, 4, 5, 1, 1, 5, 5, 4, 1, 2, 5, 1, 3, 1, 3, 4, 4, 1, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 5, 1, 4, 1, 2, 4, 1, 2, 4, 5, 1, 3, 4, 2, 4, 1, 3, 4, 4, 1, 5, 5, 4, 2, 1, 5, 1, 2, 3, 3, 2, 5, 3, 4, 1, 2, 3, 4, 5, 5, 2, 2, 2, 1, 3, 4, 5, 1, 2, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 5, 2, 3, 5, 1, 2, 3, 4, 5, 4, 5, 1, 1, 3, 4, 5, 3, 4, 5, 1, 4, 4, 1, 2, 3, 4, 1, 2, 4, 3, 5, 5, 1, 1, 2, 3, 4, 5, 5, 1, 4, 1, 4, 5, 1, 3, 1, 2, 3, 5, 3, 1, 5, 5, 4, 2, 1, 3, 4, 1, 2, 3, 4, 5, 3, 1, 2, 4, 4, 1, 3, 1, 1, 1, 2, 4, 1, 2, 3, 4, 4, 1, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 3, 4, 4, 1, 3, 1, 3, 4, 5, 1, 1, 5, 1, 3, 4, 5, 3, 4, 1, 2, 3, 5, 1, 1, 2, 2, 5, 1, 2, 5, 1, 2, 3, 5, 5, 2, 5, 1, 3, 1, 3, 4, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5, 1, 2, 5, 2, 2, 3, 4, 5, 2, 4, 3, 1, 1, 4, 1, 2, 3, 4, 5, 3, 1, 2, 3, 4, 5, 1, 3, 1, 2, 3, 4, 4, 3, 2, 3, 1, 2, 3, 4, 1, 1, 4, 5, 1, 5, 5, 5, 4, 3, 5, 5, 1, 4, 1, 2, 1, 2, 4, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 5, 1, 2, 3, 4, 5, 1, 3, 5, 1, 3, 4, 3, 4, 1, 3, 4, 5, 2, 5, 1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 4], \"Freq\": [0.2275091937456275, 0.7725312183230648, 0.9998663409243406, 0.4495250132372862, 0.5494194606233498, 0.9972980586900414, 0.987061478594432, 0.15936352054331113, 0.3947592550862936, 0.14294056231938213, 0.0456193283998028, 0.2572930121748878, 0.9942970144207509, 0.1458728246554686, 0.6951957138985846, 0.15901452057037568, 0.9931394036712324, 0.14650806003263636, 0.853494628794777, 0.8197024312056687, 0.1796608068395986, 0.9968428670277433, 0.894485915462301, 0.10203261393106094, 0.057126956724836676, 0.9425947859598052, 0.9962680152725506, 0.9786633855880057, 0.468026323511442, 0.14625822609732564, 0.12480701960305121, 0.13390753144910703, 0.12675712928434887, 0.9999116560901258, 0.9983162210587824, 0.14173551207368087, 0.8576077684864346, 1.000486903585736, 0.999807714954941, 0.9993535302288455, 0.5513024722347107, 0.44848018574649084, 0.03681255800272127, 0.9632619344045399, 0.9971690875098018, 0.126338924039978, 0.28248815599950144, 0.03548846180898259, 0.556459081164847, 0.9891513272439507, 0.992649336865892, 0.38249091325250323, 0.1833777637903615, 0.43393352025640003, 0.40759879596923854, 0.17985734836464967, 0.4122704154072814, 0.8464592147928178, 0.15390167541687597, 0.9975844224259919, 0.9998884563785616, 0.9934669016778007, 0.08422351261774831, 0.1704765074672496, 0.3683510250631643, 0.19685977648003822, 0.17960917751013797, 0.9964935696476255, 0.9901207119958787, 0.9771589774644022, 0.07111116473282368, 0.699883568686212, 0.10217551564242561, 0.02432750372438705, 0.10217551564242561, 1.0003459862326225, 0.8363293546897406, 0.12595321606773202, 0.03526690049896496, 0.7499408770297286, 0.2503208649486151, 0.9956585721872605, 0.2229932580788558, 0.7752905135532313, 0.997448835240023, 0.2761805179706036, 0.5884769498296708, 0.135965793462451, 0.9994833407619615, 0.995313210619752, 0.285908851710608, 0.5697307774963211, 0.14399788881775147, 0.045757559570505216, 0.9517572390665084, 0.9999330725747337, 0.9990572956097886, 0.9860487114557388, 0.9909726825436558, 0.9936284587122572, 0.9997599335791891, 0.9996709788619824, 0.9987700769966814, 0.994440473076659, 0.9010544870118464, 0.09731388459727941, 0.07719907073648863, 0.9263888488378635, 0.15879344475057589, 0.7016454535490563, 0.13848265530573478, 0.06103669469441454, 0.851607216450641, 0.08428876886371532, 0.9968685991752269, 0.9942970046587377, 0.8941476319701497, 0.10624377756299307, 0.08514081339708751, 0.9152637440186908, 0.21969559765268332, 0.6863091095956425, 0.09431957595560227, 0.9968685966774083, 0.9937979597702202, 0.13213117509708125, 0.7727314257909662, 0.04247073485263326, 0.054268161200586945, 0.9966359815190067, 0.5954370458414746, 0.10327363396708503, 0.30013899871684085, 0.9891513327725212, 0.9849819311400336, 0.8643885295461381, 0.13387835811152307, 0.9983774258090634, 0.9906247193679044, 0.996733460668463, 0.9959347163987854, 0.9891513294315328, 0.982668092963985, 0.45945357877534193, 0.08511252381337013, 0.28229628939004664, 0.11053074359473858, 0.06277530036913724, 0.11712943779323282, 0.6991163318283584, 0.18301474655192626, 0.9874635925174251, 0.9924713155426662, 0.6271668934183194, 0.3729575915979908, 0.5060408199207573, 0.04166146236369108, 0.3354145759726467, 0.11702359810438069, 1.0011571377180903, 0.9984782940661686, 0.9976139862393452, 0.9972604105910401, 0.033986258645119304, 0.033986258645119304, 0.4201937432487478, 0.5113387096152041, 0.16914011120208602, 0.10430306857461971, 0.7273024781689699, 0.9982393246908904, 0.7989551920516655, 0.12496478644910666, 0.07784691614862382, 0.8177821817422176, 0.06052432670381301, 0.12104865340762602, 0.9870614691192361, 1.0007170944003299, 0.17829199034018162, 0.8211245793648732, 0.9922794019571657, 0.9980362702387137, 0.14235176271249983, 0.6284309524624992, 0.2291516180249997, 0.046829287110925405, 0.9459515996406932, 0.10578614465217338, 0.023633074869102565, 0.22845305706799146, 0.6414691750184982, 0.11064533561448837, 0.04539295820081574, 0.8426067866026422, 0.34685019740061696, 0.15945246041639738, 0.49315193943215685, 0.09395451818946017, 0.3106025836616272, 0.5957821800484592, 0.9949152723683496, 0.22101368970767238, 0.11206327928839727, 0.5198491011433984, 0.14630483684874088, 0.2305199017267174, 0.7651298865822961, 0.9991295442404166, 0.9907211392552278, 0.9981443652870979, 0.18833818562429358, 0.06519398733148624, 0.7026463079060183, 0.04346265822099083, 0.10036442202813477, 0.2435198767039239, 0.6566478464476414, 0.999226138205228, 0.8297664075828667, 0.048433060934648754, 0.12208333541378405, 0.9979786821427818, 0.08103946212858226, 0.9218238817126233, 0.1658651363283788, 0.5779683299837644, 0.2559647165561401, 0.9701507886529148, 0.028211169944672806, 0.9948466668575219, 0.9948183047005978, 0.7438218279863434, 0.14930730123813463, 0.1058724499688591, 0.9988994790333185, 0.3933297917595175, 0.49208246383589205, 0.06261339633778644, 0.05189768085715856, 0.48348511782083126, 0.05934248862141986, 0.3672987716778408, 0.08995071959457325, 0.2757673127117129, 0.5305669161235693, 0.1717674745844246, 0.02197415934624963, 0.9976321268427141, 0.03525949044686425, 0.7983756051182834, 0.16622331210664576, 0.3361890036636517, 0.18752250958123287, 0.476408537855024, 0.625651992078483, 0.16483743061812434, 0.2094888250679808, 0.31642828423061203, 0.16965130901520764, 0.3354902290637814, 0.1782291841901339, 0.4044757660995503, 0.5954211785221807, 0.9676811412810856, 0.028972489259912745, 0.9826680931058349, 0.9979242507992412, 0.9885874080062808, 0.9985194648226815, 0.9915654014169588, 0.9987717678455553, 0.40356348002815523, 0.2180710295715777, 0.19106896399878476, 0.1347168271512169, 0.052536627581847115, 0.9989356431722972, 0.09710849568396245, 0.9017217456367941, 0.9991883702391287, 0.9955546788329642, 0.15955000828424512, 0.34134032075356685, 0.49895638954345745, 0.20665396876812606, 0.7921735469444833, 0.38055362732618214, 0.5444844206359221, 0.07464705766782803, 0.9856404686159306, 0.45684403862358297, 0.21933831146331093, 0.1608480950730947, 0.16306363357272408, 0.4000908590131255, 0.31246288989209514, 0.09685196587061251, 0.048425982935306255, 0.14181895002482547, 0.2534542250579346, 0.03127732990076639, 0.1962922083427408, 0.04421967330798007, 0.4745525915978349, 0.8690909230437359, 0.13036363845656038, 0.9713382423788285, 0.028636539168742148, 0.33482982459283067, 0.6630293556293677, 0.3243927544064319, 0.1871105542657789, 0.48811448938898844, 0.03650066853838156, 0.38712830267980447, 0.4501749119733726, 0.12609321858713632, 0.8622889139847368, 0.06576779852425958, 0.07307533169362176, 0.9998256646575424, 1.0014104228888598, 0.47062623911358287, 0.29129670506063154, 0.2384991772683921, 0.9906468849020577, 0.05381571693969648, 0.9417750464446885, 0.9955079085999593, 0.9983023045788698, 0.9997165969275281, 0.9998884506633131, 0.996157822768926, 0.20417620300337577, 0.39280599461055543, 0.40213384144827313, 0.9873912137618167, 0.8873496535168505, 0.11240897722094459, 0.14478018885837546, 0.8534411132704237, 0.1892421593839962, 0.08644394934824517, 0.5560448633751987, 0.08644394934824517, 0.0829394649152082, 0.9872952234365906, 0.99702589250225, 0.9920013936224236, 0.9966919172053751, 0.06574998328426063, 0.03238432012508359, 0.2688879913416031, 0.6329662569902702, 0.3108302282240691, 0.35698467091178604, 0.3319131217974706, 0.22682449170500424, 0.1462174409338044, 0.3745415986996682, 0.01462174409338044, 0.23769707064623585, 0.06370670785539127, 0.6482587975014815, 0.09039465303805519, 0.1179434996782244, 0.08006383554799175, 0.3399930112102098, 0.02729618687561644, 0.48442093670397784, 0.14822866037518295, 0.3497787818108969, 0.09696837515549617, 0.5541050008885495, 0.3839351089269073, 0.06792986775735115, 0.34465338061050166, 0.06217521966004331, 0.1413641815208228, 0.9870614690410234, 0.9984740768006031, 0.9986631536306217, 0.1735087357605733, 0.04406571066935195, 0.4186242513588435, 0.3649191664805708, 0.9885874115430898, 0.9942970053738471, 0.9973301817628831, 0.11298661846606421, 0.8877520022333617, 0.9936284577163446, 0.7285700319962863, 0.03113547145283275, 0.21511780276502626, 0.0249083771622662, 0.10257089358872311, 0.41505431359157724, 0.48184466290516437, 0.9395089522905324, 0.05722897171312888, 1.0003331136973228, 0.9979132662796789, 0.30467045113701935, 0.2761222214605859, 0.060840489474366384, 0.18205346465791172, 0.17596941571047509, 0.9879050824941326, 0.09372613873959494, 0.9021140853686013, 0.8222241278449679, 0.08997402072377594, 0.08766699455137143, 0.9978749640189314, 0.9945543895798153, 0.0757752029616258, 0.6150584655976121, 0.2479915733289572, 0.061013799787283114, 0.9846833354857284, 0.9991129283716637, 1.0001235280488545, 0.9960786835136102, 0.9826681007374002, 0.9966745417474421, 0.2391619384034255, 0.7294439121304478, 0.029895242300428188, 0.4037820489249065, 0.24493962374920208, 0.051105823752183144, 0.11510320664906114, 0.18462554346509405, 0.9979983307359731, 0.22134687880993326, 0.5469017383654364, 0.2319958695430572, 0.9898137987195444, 0.21823828513820392, 0.7806215583789602, 1.0000411406590475, 0.9970028247641061, 0.511357763174512, 0.4346892677329414, 0.05345693260146205, 0.20450146924518736, 0.12742014622200135, 0.12742014622200135, 0.541142349387265, 0.9962421291560767, 0.9984201144497592, 0.9937011411644499, 0.05127437112498135, 0.1205223390099024, 0.25383570393485394, 0.022935632675260477, 0.5514475913893155, 0.1759443678637649, 0.3989471100965436, 0.15256504811355426, 0.27245899555053205, 0.3334745350740059, 0.6423080946499325, 0.0229982437982073, 0.9997423081902519, 0.999378030679456, 0.99578815141794, 0.47987377943389276, 0.1808366453024564, 0.3394475576627115, 0.997687116332895, 0.9988994828359857, 0.05437377359067175, 0.9424787422383103, 0.2527897678897211, 0.1415769032899017, 0.45436308497689376, 0.1514543616589646, 0.986756123777416, 0.9826680945803796, 0.14053412914799343, 0.8581950819970798, 0.025049847119605475, 0.9745583379389369, 0.9966133971642891, 0.997348872642841, 0.9946990176970015, 1.0003459880533547, 0.9964533773863181, 0.03743940044296873, 0.33892509874687476, 0.6246468389695309, 0.11122830100609078, 0.3940982441738648, 0.01863215194518272, 0.4759667905996677, 0.9981648994101957, 0.09826482189473221, 0.9007608673683786, 0.9562300717055049, 0.04213895231244598, 0.12814553762070605, 0.11055693441786404, 0.7613352529230183, 0.16143747506386782, 0.398728448685142, 0.06103663440770893, 0.3786040511086873, 0.9993124594774986, 0.013344708385859336, 0.410967593438594, 0.12405636314261827, 0.32126149817809513, 0.13048159310618018, 0.08384787707252599, 0.4498896597546877, 0.2825462076141422, 0.18354934434784048, 0.01972490567068827, 0.20028365757929628, 0.17297224972757408, 0.4885707404585864, 0.11986673446033641, 0.9995716823562726, 0.739026588915257, 0.2602616633762009, 0.9979177029110959, 0.996596292097088, 0.9873912202104521, 0.18586447642992904, 0.8131570843809396, 0.1922296127710161, 0.8038692897697037, 0.9934013070819631, 0.9997623644635693, 0.14312325659564884, 0.8522339370013636, 0.25925160130453734, 0.3250932778263246, 0.01920382231885462, 0.17191993314022228, 0.22450182758470516, 0.9907211492479651, 0.5247432516666404, 0.09588456275992858, 0.18727453664048552, 0.15581241448488395, 0.03633126010825419, 0.15357683474773967, 0.8446725911125682, 0.4599372694534525, 0.061495790901854364, 0.4074096147247852, 0.07046392707504147, 0.9961468796484083, 0.9867561234033118, 0.9599773126269958, 0.03770303644758594, 0.4294662403463404, 0.13841681962382052, 0.08080549469931143, 0.35090534272200985, 0.9992422805162753, 0.20625708559911138, 0.7596297542796542, 0.030183963746211424, 0.10074111015316217, 0.898274898865696, 0.9990582224652079, 0.9935264005204978, 0.9945660114815684, 0.9867561216733743, 0.991110877026184, 0.9890551000698292, 0.9298173057893846, 0.06927446814656012, 0.057944576097052315, 0.942361790209956, 0.05565953637847903, 0.15233136272004788, 0.16697860913543708, 0.6247050596163501, 0.09726573812953733, 0.32061669235291934, 0.04503043431923025, 0.5385639944579937, 1.0004965718200713, 0.20087050116355143, 0.7987969783880005, 0.12302798541017117, 0.876260549145913, 0.11811093713239353, 0.8787453722650078, 0.5085625686184567, 0.17730497715439394, 0.03978550706879084, 0.1011935723271419, 0.17298046551648188, 0.714140714356276, 0.05158911249219744, 0.2343619681788398, 0.3327000548767451, 0.39167069952421046, 0.2754897279799503, 0.20647015548342995, 0.7888218760777196, 0.6419910959614114, 0.23160438272025602, 0.12596027832154275, 1.0004965729790323, 0.07298007202887029, 0.9284686941450719, 0.20670265484587064, 0.0943456100956256, 0.6990152020721352, 0.33640026865392486, 0.03737780762821387, 0.04093759883090091, 0.5375284716057424, 0.048057181236274976, 0.22189077385290298, 0.12766318495646473, 0.6474347237077854], \"Term\": [\"absolutely\", \"absolutely\", \"activities\", \"amazing\", \"amazing\", \"amsterdam\", \"arc\", \"area\", \"area\", \"area\", \"area\", \"area\", \"argonaut\", \"arrived\", \"arrived\", \"arrived\", \"arriving\", \"asked\", \"asked\", \"attentive\", \"attentive\", \"attractions\", \"awesome\", \"awesome\", \"bags\", \"bags\", \"bali\", \"balinese\", \"bar\", \"bar\", \"bar\", \"bar\", \"bar\", \"barcelona\", \"bath\", \"bathroom\", \"bathroom\", \"bavaro\", \"beach\", \"beaches\", \"beautiful\", \"beautiful\", \"bed\", \"bed\", \"bedroom\", \"beds\", \"beds\", \"beds\", \"beds\", \"bellman\", \"berlin\", \"best\", \"best\", \"best\", \"big\", \"big\", \"big\", \"birthday\", \"birthday\", \"blocks\", \"boat\", \"bonus\", \"booked\", \"booked\", \"booked\", \"booked\", \"booked\", \"boston\", \"bourbon\", \"bravo\", \"breakfast\", \"breakfast\", \"breakfast\", \"breakfast\", \"breakfast\", \"bridge\", \"bring\", \"bring\", \"bring\", \"buffet\", \"buffet\", \"cafes\", \"called\", \"called\", \"calls\", \"came\", \"came\", \"came\", \"cana\", \"canal\", \"car\", \"car\", \"car\", \"card\", \"card\", \"carpet\", \"casino\", \"castle\", \"catalunya\", \"celebrate\", \"central\", \"centre\", \"cereal\", \"channels\", \"charge\", \"charge\", \"charged\", \"charged\", \"check\", \"check\", \"check\", \"checked\", \"checked\", \"checked\", \"checkout\", \"chic\", \"city\", \"city\", \"class\", \"class\", \"clean\", \"clean\", \"clean\", \"clearly\", \"clerk\", \"close\", \"close\", \"close\", \"close\", \"closet\", \"comfortable\", \"comfortable\", \"comfortable\", \"computers\", \"condado\", \"convenient\", \"convenient\", \"country\", \"cramped\", \"credit\", \"croissants\", \"curtain\", \"david\", \"day\", \"day\", \"day\", \"day\", \"day\", \"decided\", \"decided\", \"decided\", \"delightful\", \"designed\", \"desk\", \"desk\", \"did\", \"did\", \"did\", \"did\", \"didn__\\u00e7_\\u00e9_\", \"distance\", \"district\", \"dominican\", \"door\", \"door\", \"door\", \"door\", \"double\", \"double\", \"double\", \"downtown\", \"drink\", \"drink\", \"drink\", \"drinks\", \"drinks\", \"drinks\", \"dua\", \"duomo\", \"easy\", \"easy\", \"elevators\", \"entertainment\", \"excellent\", \"excellent\", \"excellent\", \"exceptional\", \"exceptional\", \"experience\", \"experience\", \"experience\", \"experience\", \"fabulous\", \"fabulous\", \"fabulous\", \"family\", \"family\", \"family\", \"fantastic\", \"fantastic\", \"fantastic\", \"fare\", \"feel\", \"feel\", \"feel\", \"feel\", \"finally\", \"finally\", \"fish\", \"fix\", \"flat\", \"flight\", \"flight\", \"flight\", \"flight\", \"floor\", \"floor\", \"floor\", \"florence\", \"food\", \"food\", \"food\", \"francisco\", \"friday\", \"friday\", \"friendly\", \"friendly\", \"friendly\", \"fun\", \"fun\", \"furnished\", \"furnishings\", \"given\", \"given\", \"given\", \"golf\", \"good\", \"good\", \"good\", \"good\", \"got\", \"got\", \"got\", \"got\", \"great\", \"great\", \"great\", \"great\", \"grounds\", \"helpful\", \"helpful\", \"helpful\", \"home\", \"home\", \"home\", \"hotel\", \"hotel\", \"hotel\", \"hotels\", \"hotels\", \"hotels\", \"hotels\", \"hours\", \"hours\", \"ideal\", \"ideal\", \"impeccable\", \"inclusive\", \"informed\", \"iron\", \"it__\\u00e7_\\u00e9_\", \"juan\", \"just\", \"just\", \"just\", \"just\", \"just\", \"kids\", \"king\", \"king\", \"kitchen\", \"kuta\", \"large\", \"large\", \"large\", \"later\", \"later\", \"left\", \"left\", \"left\", \"library\", \"like\", \"like\", \"like\", \"like\", \"little\", \"little\", \"little\", \"little\", \"little\", \"lobby\", \"lobby\", \"lobby\", \"lobby\", \"lobby\", \"located\", \"located\", \"location\", \"location\", \"love\", \"love\", \"loved\", \"loved\", \"loved\", \"lovely\", \"lovely\", \"lovely\", \"lovely\", \"lunch\", \"lunch\", \"lunch\", \"madrid\", \"magnificent\", \"make\", \"make\", \"make\", \"mandarin\", \"marble\", \"marble\", \"mattress\", \"memorable\", \"metro\", \"mexican\", \"microwave\", \"minutes\", \"minutes\", \"minutes\", \"mistake\", \"modern\", \"modern\", \"moment\", \"moment\", \"morning\", \"morning\", \"morning\", \"morning\", \"morning\", \"motel\", \"museum\", \"museums\", \"neighborhood\", \"new\", \"new\", \"new\", \"new\", \"nice\", \"nice\", \"nice\", \"night\", \"night\", \"night\", \"night\", \"night\", \"nights\", \"nights\", \"nights\", \"nights\", \"nights\", \"no\", \"no\", \"no\", \"no\", \"noise\", \"noise\", \"noise\", \"not\", \"not\", \"not\", \"not\", \"not\", \"nusa\", \"nyc\", \"ocean\", \"old\", \"old\", \"old\", \"old\", \"omni\", \"oriental\", \"orleans\", \"outstanding\", \"outstanding\", \"owners\", \"people\", \"people\", \"people\", \"people\", \"perfect\", \"perfect\", \"perfect\", \"phone\", \"phone\", \"pillows\", \"pizza\", \"place\", \"place\", \"place\", \"place\", \"place\", \"plasma\", \"pleasure\", \"pleasure\", \"pool\", \"pool\", \"pool\", \"pools\", \"port\", \"price\", \"price\", \"price\", \"price\", \"priceline\", \"punta\", \"quarter\", \"queen\", \"raffles\", \"ramblas\", \"ready\", \"ready\", \"ready\", \"really\", \"really\", \"really\", \"really\", \"really\", \"receptionist\", \"recommend\", \"recommend\", \"recommend\", \"recommendation\", \"reservation\", \"reservation\", \"resort\", \"resorts\", \"restaurants\", \"restaurants\", \"restaurants\", \"return\", \"return\", \"return\", \"return\", \"ritz\", \"riu\", \"robes\", \"room\", \"room\", \"room\", \"room\", \"room\", \"rooms\", \"rooms\", \"rooms\", \"rooms\", \"said\", \"said\", \"said\", \"san\", \"sand\", \"saturday\", \"say\", \"say\", \"say\", \"screen\", \"seafood\", \"separate\", \"separate\", \"service\", \"service\", \"service\", \"service\", \"shampoo\", \"shangri\", \"shopping\", \"shopping\", \"shower\", \"shower\", \"shows\", \"sick\", \"sights\", \"sightseeing\", \"sink\", \"size\", \"size\", \"size\", \"small\", \"small\", \"small\", \"small\", \"sofa\", \"space\", \"space\", \"spanish\", \"spanish\", \"special\", \"special\", \"special\", \"staff\", \"staff\", \"staff\", \"staff\", \"station\", \"stay\", \"stay\", \"stay\", \"stay\", \"stay\", \"stayed\", \"stayed\", \"stayed\", \"stayed\", \"staying\", \"staying\", \"staying\", \"staying\", \"staying\", \"steak\", \"street\", \"street\", \"stylish\", \"subway\", \"suggested\", \"suite\", \"suite\", \"superb\", \"superb\", \"supermarket\", \"swim\", \"thanks\", \"thanks\", \"the\", \"the\", \"the\", \"the\", \"the\", \"ticket\", \"time\", \"time\", \"time\", \"time\", \"time\", \"told\", \"told\", \"took\", \"took\", \"took\", \"took\", \"top\", \"toronto\", \"train\", \"train\", \"trip\", \"trip\", \"trip\", \"trip\", \"tropical\", \"truly\", \"truly\", \"truly\", \"tub\", \"tub\", \"tv\", \"twin\", \"ubud\", \"unable\", \"unit\", \"upstairs\", \"vacation\", \"vacation\", \"value\", \"value\", \"view\", \"view\", \"view\", \"view\", \"visit\", \"visit\", \"visit\", \"visit\", \"waikiki\", \"walk\", \"walk\", \"walking\", \"walking\", \"wall\", \"wall\", \"want\", \"want\", \"want\", \"want\", \"want\", \"water\", \"water\", \"water\", \"way\", \"way\", \"way\", \"welcome\", \"welcome\", \"went\", \"went\", \"went\", \"westin\", \"window\", \"window\", \"wonderful\", \"wonderful\", \"wonderful\", \"year\", \"year\", \"year\", \"year\", \"year\", \"years\", \"years\", \"years\"]}, \"R\": 30, \"lambda.step\": 0.01, \"plot.opts\": {\"xlab\": \"PC1\", \"ylab\": \"PC2\"}, \"topic.order\": [2, 3, 1, 5, 4]};\n",
       "\n",
       "function LDAvis_load_lib(url, callback){\n",
       "  var s = document.createElement('script');\n",
       "  s.src = url;\n",
       "  s.async = true;\n",
       "  s.onreadystatechange = s.onload = callback;\n",
       "  s.onerror = function(){console.warn(\"failed to load library \" + url);};\n",
       "  document.getElementsByTagName(\"head\")[0].appendChild(s);\n",
       "}\n",
       "\n",
       "if(typeof(LDAvis) !== \"undefined\"){\n",
       "   // already loaded: just create the visualization\n",
       "   !function(LDAvis){\n",
       "       new LDAvis(\"#\" + \"ldavis_el1467229661875813287835259874\", ldavis_el1467229661875813287835259874_data);\n",
       "   }(LDAvis);\n",
       "}else if(typeof define === \"function\" && define.amd){\n",
       "   // require.js is available: use it to load d3/LDAvis\n",
       "   require.config({paths: {d3: \"https://d3js.org/d3.v5\"}});\n",
       "   require([\"d3\"], function(d3){\n",
       "      window.d3 = d3;\n",
       "      LDAvis_load_lib(\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js\", function(){\n",
       "        new LDAvis(\"#\" + \"ldavis_el1467229661875813287835259874\", ldavis_el1467229661875813287835259874_data);\n",
       "      });\n",
       "    });\n",
       "}else{\n",
       "    // require.js not available: dynamically load d3 & LDAvis\n",
       "    LDAvis_load_lib(\"https://d3js.org/d3.v5.js\", function(){\n",
       "         LDAvis_load_lib(\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js\", function(){\n",
       "                 new LDAvis(\"#\" + \"ldavis_el1467229661875813287835259874\", ldavis_el1467229661875813287835259874_data);\n",
       "            })\n",
       "         });\n",
       "}\n",
       "</script>"
      ],
      "text/plain": [
       "PreparedData(topic_coordinates=                x           y  topics  cluster       Freq\n",
       "topic                                                    \n",
       "1      -83.110329 -142.085129       1        1  29.862520\n",
       "2       89.410652 -180.259003       2        1  21.059909\n",
       "0     -126.839188   35.645428       3        1  20.564233\n",
       "4       49.309772   55.002243       4        1  14.424794\n",
       "3      201.813751  -35.339371       5        1  14.088544, topic_info=           Term          Freq         Total Category  logprob  loglift\n",
       "12269     hotel  13146.000000  13146.000000  Default  30.0000  30.0000\n",
       "20790      room   9068.000000   9068.000000  Default  29.0000  29.0000\n",
       "14564  location   3038.000000   3038.000000  Default  28.0000  28.0000\n",
       "3193      beach   2854.000000   2854.000000  Default  27.0000  27.0000\n",
       "11271     great   5961.000000   5961.000000  Default  26.0000  26.0000\n",
       "...         ...           ...           ...      ...      ...      ...\n",
       "19701    really    400.966291   2171.963816   Topic5  -5.3280   0.2703\n",
       "7569        did    440.996846   3768.470694   Topic5  -5.2329  -0.1856\n",
       "21657   service    413.565184   2733.496715   Topic5  -5.2971   0.0713\n",
       "18390     place    375.604980   2136.734946   Topic5  -5.3934   0.2213\n",
       "14374      like    368.189026   2256.787684   Topic5  -5.4133   0.1467\n",
       "\n",
       "[390 rows x 6 columns], token_table=       Topic      Freq        Term\n",
       "term                              \n",
       "1108       1  0.227509  absolutely\n",
       "1108       4  0.772531  absolutely\n",
       "1286       1  0.999866  activities\n",
       "1831       1  0.449525     amazing\n",
       "1831       4  0.549419     amazing\n",
       "...      ...       ...         ...\n",
       "27247      4  0.537528        year\n",
       "27247      5  0.048057        year\n",
       "27250      1  0.221891       years\n",
       "27250      3  0.127663       years\n",
       "27250      4  0.647435       years\n",
       "\n",
       "[625 rows x 3 columns], R=30, lambda_step=0.01, plot_opts={'xlab': 'PC1', 'ylab': 'PC2'}, topic_order=[2, 3, 1, 5, 4])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Topic Model Visualization\n",
    "#prepare to display result in the Jupyter notebook\n",
    "pyLDAvis.enable_notebook()\n",
    "\n",
    "#run the visualization [mds is a function to use for visualizing the \"distance\" between topics]\n",
    "pyLDAvis.lda_model.prepare(lda_positive_corpus, bow_positive_corpus, bow_vectorizer, mds='tsne')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f3e8fb1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

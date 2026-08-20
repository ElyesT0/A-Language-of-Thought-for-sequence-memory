import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import math
import random
import seaborn as sns
from datetime import datetime
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportion_confint
from modules.params import *
from IPython.display import Markdown
import os
import zlib
import warnings
import collections
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from scipy.stats import friedmanchisquare
from scipy.stats import shapiro
from scipy.stats import kstest, norm
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf



"""
This file contains the constant necessary for the data analysis
"""

# ---------------------------------------
# ************** Constants **************
# ---------------------------------------
#SOA: 400ms
#delay before presentation and between presentation and reproduction: 750ms

experimenter_id=['BDT45D782PQS']

# Labels of columns that will need to be turned into ints/arrays of ints
# label_int_col_old is used for pre_processing
# label_int_col is used in plotting 
label_int_col_old=['sequences_structure',
    'seq',
    'confidence',
    'counter',
    'click_timings_before',
    'click_timings_after',
    'interclick_timings_before',
    'interclick_time',
    'response_sequences_before',
    'response_sequences_after',
    'score'
]

label_int_col=['sequences_structure',
    'seq',
    'confidence',
    'counter',
    'click_timings_before',
    'click_timings_after',
    'interclick_timings_before',
    'interclick_time',
    'sequences_response_before',
    'sequences_response',
    'score',
    "comparable_temp",
    "geom_dist_point",
    'response_structure'
]

label_int_col_old_exp1=['sequences_structure',
    'seq',
    'confidence',
    'click_timings_after',
    'interclick_time',
    'sequences_response',
    'score',
    'geom_dist_point'
]

seq_name_list=[
    "Repetition-2", 
    "control Repetition-2", 
    "Repetition-3", 
    "control Repetition-3", 
    "Repetition-4", 
    "control Repetition-4",
    "Repetition-Nested",
    "control NoLocal nested",
    "control NoGlobal nested",
    "play 4 tokens",
    "control play 4 tokens",
    "sub-programs 1",
    "control sub-programs 1",
    "sub-programs 2",
    "control sub-programs 2",
    "index i",
    "control index i",
    "play",
    "control play",
    "Insertion",
    "Suppression",
    "Mirror-Rep",
    "control Mirror-Rep",
    "Mirror-NoRep",
    "control Mirror-NoRep",
    
]

seq_name_list_exp1_only=[
    "Repetition-2", 
    "control Repetition-2", 
    "Repetition-3", 
    "control Repetition-3", 
    "Repetition-4", 
    "control Repetition-4",
    "Repetition-Nested",
    "control NoLocal nested",
    "control NoGlobal nested",
]

seq_name_list_exp2_only=[
   "play 4 tokens",
    "control play 4 tokens",
    "sub-programs 1",
    "control sub-programs 1",
    "sub-programs 2",
    "control sub-programs 2",
    "index i",
    "control index i",
    "play",
    "control play",
    "Insertion",
    "Suppression",
    "Mirror-Rep",
    "control Mirror-Rep",
    "Mirror-NoRep",
    "control Mirror-NoRep",
]

pairs_for_stat_test_exp1 = [
  ['Repetition-2','control Repetition-2'],
  ['Repetition-3','control Repetition-3'],
  ['Repetition-4','control Repetition-4'],
  ['Repetition-Nested','Repetition-3'],
  ['Repetition-Nested','control Repetition-3'],
  ['Repetition-Nested','control NoLocal nested'],
  ['Repetition-Nested','control NoGlobal nested'],
  ['control NoLocal nested','control Repetition-3'],
  ['control NoGlobal nested','control Repetition-3'],
  ['control NoLocal nested','control NoGlobal nested'],
]

group_structured_exp1 = [
  "Repetition-2",
  "Repetition-3",
  "Repetition-4",
]

group_control_exp1 = [
  "control Repetition-2",
  "control Repetition-3",
  "control Repetition-4",
]

# Create a list that contains the sequences expressions
list_seq_expression=[
  '121212121212',
 '122221121112',
 '123123123123',
 '123132231213',
 '123412341234',
 '123432411423',
 '112233112233',
 '123132123132',
 '112233113322',
 '121314121314',
 '121324121324',
 '123412321231',
 '123413231231',
 '123412351236',
 '123413251236',
 '121122111222',
 '111222121122',
 '111211131114',
 '111211311114',
 '123123412345',
 '123451234123',
 '123443211234',
 '123442311234',
 '123432141234',
 '123431241234']

list_seq_expression_letters=[
                    "ABABABABABAB",
                     "ABBBBAABAAAB",
                     "ABCABCABCABC",
                     "ABCACBBCABAC",
                     "ABCDABCDABCD",
                     "ABCDCBDAADBC",
                     "AABBCCAABBCC",
                     "ABCACBABCACB",
                     "AABBCCAACCBB",
                        "ABACADABACAD",
                     "ABACBDABACBD",
                     "ABCDABCBABCA",
                     "ABCDACBCABCA",
                     "ABCDABCEABCF",
                     "ABCDACBEABCF",
                     "ABAABBAAABBB",
                     "AAABBBABAABB",
                     "AAABAAACAAAD",
                     "AAABAACAAAAD",
                     "ABCABCDABCDE",
                     "ABCDEABCDABC",
                     "ABCDDCBAABCD",
                     "ABCDDBCAABCD",
                     "ABCDCBADABCD",
                     "ABCDCABDABCD"
                     
                    ]

dict_expressions=dict(zip(seq_name_list,list_seq_expression))

alpha_seq_expression=[
                    "ABABABABABAB",
                     "ABBBBAABAAAB",
                     "ABCABCABCABC",
                     "ABCACBBCABAC",
                     "ABCDABCDABCD",
                     "ABCDCBDAADBC",
                     "AABBCCAABBCC",
                     "ABCACBABCACB",
                     "AABBCCAACCBB",
                    "ABACADABACAD",
                     "ABACBDABACBD",
                     "ABCDABCBABCA",
                     "ABCDACBCABCA",
                     "ABCDABCEABCF",
                     "ABCDACBEABCF",
                     "ABAABBAAABBB",
                     "AAABBBABAABB",
                     "AAABAAACAAAD",
                     "AAABAACAAAAD",
                     "ABCABCDABCDE",
                     "ABCDEABCDABC",
                     "ABCDDCBAABCD",
                     "ABCDDBCAABCD",
                     "ABCDCBADABCD",
                     "ABCDCABDABCD"
                     
                    ]

tested_sequences=["010203010203",
"010213010213",
"012301210120",
"012302120120",
"012301240125",
"012302140125",
"010011000111",
"000111010011",
"000100020003",
"000100200003",
"012012301234",
"012340123012",
"012332100123",
"012331200123",
"012321030123",
"012320130123",
"121212121212",
"122221121112",
"123123123123",
"123132231213",
"123412341234",
"123432411423",
"112233112233",
"123132123132",
"112233113322"]

real_mapping={'play 4 tokens': '010203010203',
 'control play 4 tokens': '010213010213',
 'sub-programs 1': '012301210120',
 'control sub-programs 1': '012302120120',
 'sub-programs 2': '012301240125',
 'control sub-programs 2': '012302140125',
 'index i': '010011000111',
 'control index i': '000111010011',
 'play': '000100020003',
 'control play': '000100200003',
 'Insertion': '012012301234',
 'Suppression': '012340123012',
 'Mirror-Rep': '012332100123',
 'control Mirror-Rep': '012331200123',
 'Mirror-NoRep': '012321030123',
 'control Mirror-NoRep': '012320130123',

 'Repetition-2': '010101010101',
 'control Repetition-2': '011110010001',
 'Repetition-3': '012012012012',
 'control Repetition-3': '012021120102',
 'Repetition-4': '012301230123',
 'control Repetition-4': '012321300312',
 'Repetition-Nested': '001122001122',
 'control NoLocal nested': '012021012021',
 'control NoGlobal nested': '001122002211'}

reverse_mapping = {value: key for key, value in real_mapping.items()}


## List complexities calculated with Santiago's code 

''' Parameters
set(
	BASE				        = 6,
	include_MOVE_AND_PLAY  		= True,
	include_PLAY                = False,
	include_REPEAT              = True,
	include_REPEAT_JUMP         = False,
	include_REPEAT_APPLY_NOTES  = False,
	include_REPEAT_APPLY_PEVAL  = False,
	include_REFLECT    		    = False,
	include_MIRROR  			= False,
	include_SUB		  			= False,
	with_POINTERS				= False,
	with_CHUNKS					= False
 
)

Version: music6.py
Path: /Users/et/Documents/UNICOG/PSC/code_santiago/20240326_test_elyes.py
'''

complexities_initial_version={
    "play 4 tokens":10,
    "control play 4 tokens":8,
    "sub-programs 1":13,
    "control sub-programs 1":12,
    "sub-programs 2":13,
    "control sub-programs 2":12,
    "index i":11,
    "control index i":1,
    "play":8,
    "control play":13,
    "Insertion":9,
    "Suppression":9,
    "Mirror-Rep":9,
    "control Mirror-Rep":10,
    "Mirror-NoRep":11,
    "control Mirror-NoRep":12,
    
    "Repetition-2":4, 
    "control Repetition-2":13, 
    "Repetition-3":4, 
    "control Repetition-3":15, 
    "Repetition-4":4, 
    "control Repetition-4":13,
    "Repetition-Nested":5,
    "control NoLocal nested":9,
    "control NoGlobal nested":11,
}

''' Parameters
set(
	BASE				        = 6,
	include_MOVE_AND_PLAY  		= True,
	include_PLAY                = True,
	include_REPEAT              = True,
	include_REPEAT_JUMP         = False,
	include_REPEAT_APPLY_NOTES  = False,
	include_REPEAT_APPLY_PEVAL  = False,
	include_REFLECT    		    = False,
	include_MIRROR  			= False,
	include_SUB		  			= False,
	with_POINTERS				= False,
	with_CHUNKS					= True
 
)

Version: music6.py
Path: /Users/et/Documents/UNICOG/PSC/code_santiago/20240326_test_elyes.py
'''

complexities_play_version={
    "play 4 tokens":14,
    "control play 4 tokens":18,
    "sub-programs 1":17,
    "control sub-programs 1":15,
    "sub-programs 2":17,
    "control sub-programs 2":16,
    "index i":16,
    "control index i":16,
    "play":6,
    "control play":15,
    "Insertion":15,
    "Suppression":13,
    "Mirror-Rep":19,
    "control Mirror-Rep":19,
    "Mirror-NoRep":14,
    "control Mirror-NoRep":17,
    "Repetition-2":8, 
    "control Repetition-2":15, 
    "Repetition-3":9, 
    "control Repetition-3":17, 
    "Repetition-4":9, 
    "control Repetition-4":17,
    "Repetition-Nested":10,
    "control NoLocal nested":16,
    "control NoGlobal nested":18,
}

''' Parameters
set(
	BASE				        = 6,
	include_MOVE_AND_PLAY  		= True,
	include_PLAY                = False,
	include_REPEAT              = True,
	include_REPEAT_JUMP         = False,
	include_REPEAT_APPLY_NOTES  = False,
	include_REPEAT_APPLY_PEVAL  = False,
	include_REFLECT    		    = False,
	include_MIRROR  			= True,
	include_SUB		  			= True,
	with_POINTERS				= False,
	with_CHUNKS					= True
 
)

Version: music6.py
Path: /Users/et/Documents/UNICOG/PSC/code_santiago/20240326_test_elyes.py
'''

complexities_allOperationsButPlay_version={
    "play 4 tokens":13,
    "control play 4 tokens":18,
    "sub-programs 1":12,
    "control sub-programs 1":14,
    "sub-programs 2":12,
    "control sub-programs 2":17,
    "index i":18,
    "control index i":18,
    "play":7,
    "control play":15,
    "Insertion":12,
    "Suppression":11,
    "Mirror-Rep":13,
    "control Mirror-Rep":18,
    "Mirror-NoRep":13,
    "control Mirror-NoRep":16,
    "Repetition-2":8, 
    "control Repetition-2":16, 
    "Repetition-3":9, 
    "control Repetition-3":20, 
    "Repetition-4":9, 
    "control Repetition-4":18,
    "Repetition-Nested":13,
    "control NoLocal nested":11,
    "control NoGlobal nested":22,
}

''' Parameters
set(
	BASE				        = 6,
	include_MOVE_AND_PLAY  		= True,
	include_PLAY                = True,
	include_REPEAT              = True,
	include_REPEAT_JUMP         = False,
	include_REPEAT_APPLY_NOTES  = False,
	include_REPEAT_APPLY_PEVAL  = False,
	include_REFLECT    		    = False,
	include_MIRROR  			= True,
	include_SUB		  			= True,
	with_POINTERS				= False,
	with_CHUNKS					= True
 
)

Version: music6.py
Path: /Users/et/Documents/UNICOG/PSC/code_santiago/20240326_test_elyes.py
'''

complexities_allOperations_version={
    "play 4 tokens":11,
    "control play 4 tokens":18,
    "sub-programs 1":12,
    "control sub-programs 1":14,
    "sub-programs 2":12,
    "control sub-programs 2":15,
    "index i":15,
    "control index i":15,
    "play":6,
    "control play":14,
    "Insertion":12,
    "Suppression":11,
    "Mirror-Rep":13,
    "control Mirror-Rep":17,
    "Mirror-NoRep":13,
    "control Mirror-NoRep":16,
    "Repetition-2":5, 
    "control Repetition-2":14, 
    "Repetition-3":9, 
    "control Repetition-3":16, 
    "Repetition-4":9, 
    "control Repetition-4":17,
    "Repetition-Nested":10,
    "control NoLocal nested":11,
    "control NoGlobal nested":18,
}

''' Parameters
set(
	BASE				        = 6,
	include_MOVE_AND_PLAY  		= True,
	include_PLAY                = True,
	include_REPEAT              = True,
	include_REPEAT_JUMP         = False,
	include_REPEAT_APPLY_NOTES  = False,
	include_REPEAT_APPLY_PEVAL  = False,
	include_REFLECT    		    = False,
	include_MIRROR  			= True,
	include_SUB		  			= True,
	with_POINTERS				= False,
	with_CHUNKS					= False
 
)

Version: music6.py
Path: /Users/et/Documents/UNICOG/PSC/code_santiago/20240326_test_elyes.py

'''
complexities_allOperations_noChunk={
    "play 4 tokens":11,
    "control play 4 tokens":18,
    "sub-programs 1":12,
    "control sub-programs 1":14,
    "sub-programs 2":12,
    "control sub-programs 2":15,
    "index i":15,
    "control index i":14,
    "play":6,
    "control play":12,
    "Insertion":12,
    "Suppression":11,
    "Mirror-Rep":9,
    "control Mirror-Rep":9,
    "Mirror-NoRep":13,
    "control Mirror-NoRep":16,
    "Repetition-2":5, 
    "control Repetition-2":14, 
    "Repetition-3":9, 
    "control Repetition-3":15, 
    "Repetition-4":9, 
    "control Repetition-4":11,
    "Repetition-Nested":8,
    "control NoLocal nested":11,
    "control NoGlobal nested":10,
}

complexities_post_fit_exp1={
    #These are irrelevant as the fit is only on exp1 sequences
        "play 4 tokens":10,
    "control play 4 tokens":8,
    "sub-programs 1":13,
    "control sub-programs 1":12,
    "sub-programs 2":13,
    "control sub-programs 2":12,
    "index i":11,
    "control index i":1,
    "play":8,
    "control play":13,
    "Insertion":9,
    "Suppression":9,
    "Mirror-Rep":9,
    "control Mirror-Rep":10,
    "Mirror-NoRep":11,
    "control Mirror-NoRep":12,


# Only below values are relevant. This is used in the main article.
     "Repetition-2":7.593044574599119, 
    "control Repetition-2":15.714093590000001, 
    "Repetition-3":9.135983343653503, 
    "control Repetition-3":26.263688933653505, 
    "Repetition-4":8.971814173807974, 
    "control Repetition-4":27.52233035283685,
    "Repetition-Nested":8.846920333653502,
    "control NoLocal nested":16.555304633653503,
    "control NoGlobal nested":17.266241623653503,
}
# ---------------------------------------
# ************ Sequences subsets for plotting ************
# ---------------------------------------

seq_subset1=[
    'Repetition-Nested',
 'control NoLocal nested',
 'control NoGlobal nested',
 'Repetition-2',
 'control Repetition-2',
 'Repetition-3',
 'control Repetition-3',
 'Repetition-4',
 'control Repetition-4'
 ]

seq_subset1a=[
  'Repetition-2',
 'control Repetition-2',
 'Repetition-3',
 'control Repetition-3',
 'Repetition-4',
 'control Repetition-4'
 ]

seq_subset1b=[
      'Repetition-Nested',
 'control NoGlobal nested',
 'control NoLocal nested',
 'control Repetition-3'
]

seq_subset2=['Mirror-Rep',
 'control Mirror-Rep',
 'Mirror-NoRep',
 'control Mirror-NoRep',
 'play',
 'control play',
 'play 4 tokens',
 'control play 4 tokens',
 'sub-programs 1',
 'control sub-programs 1',
 'sub-programs 2',
 'control sub-programs 2',
 'index i',
 'control index i',
 'Insertion',
 'Suppression',
 ]
# ---------------------------------------
# ************ Plot variables ************
# ---------------------------------------
plot_figsize_coef = 0.8
plot_figsize=(10,10)
y_label_pad=20
color_structure_control=['#386641','#bc4749']*3+['#386641','#bc4749','#bc4749']+['#386641','#bc4749']*8
plot_colors=['#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4','#ADE8F4',
         '#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4', 
         '#03045E', '#03045E', '#0077B6', '#0077B6', '#00B4D8', '#00B4D8', '#ADE8F4', '#ADE8F4']
figure_format_points=['o','o','s','s','v','v','s','s','s','D','D','^','^','o','o','s','s','v','v','D','D','^','^','o','o']
sub_title_size=25
title_size=15
padding_size=15
"""
'o': Circle
's': Square
'D': Diamond
'^': Upward-pointing triangle
'v': Downward-pointing triangle
"""
plot_colors2=['#FEC89A', '#FEC89A', # play-4
              '#fec5bb', '#fec5bb', # sub-programs 1
              '#d8e2dc', '#d8e2dc', # sub-programs 2
              '#ECE4DB', '#ECE4DB', # index i
              '#c997a0','#c997a0', # play
              '#9d8189', '#9d8189', # insertion - Suppression
              '#f4acb7', '#f4acb7', # Mirror 1
              '#ffe5d9', '#ffe5d9'] # Mirror 2

distinct_colors_all=[
            '#001219','#001219', # Repetition-2
            '#005F73','#005F73', # Repetition-3
            '#0A9396','#0A9396', # Repetition-4
            '#9B2226','#9B2226','#9B2226', # Repetition-Nested 
            
            '#EE9B00', '#EE9B00', # play-4
            '#fec5bb', '#fec5bb', # sub-programs 1
            '#E9D8A6', '#E9D8A6', # sub-programs 2
            '#ECE4DB', '#ECE4DB', # index i
            '#0466C8','#0466C8', # play
            '#33415C', '#33415C', # insertion - Suppression
            '#9d4edd', '#9d4edd', # Mirror 1
            '#b76935', '#b76935'] # Mirror 2

distinct_colors_base=[
            '#9B2226','#9B2226','#9B2226', # Repetition-Nested
            '#001219','#001219', # Repetition-2
            '#005F73','#005F73', # Repetition-3
            '#0A9396','#0A9396', # Repetition-4
] 

distinct_colors_ext=['#EE9B00', '#EE9B00', # play-4
            '#fec5bb', '#fec5bb', # sub-programs 1
            '#E9D8A6', '#E9D8A6', # sub-programs 2
            '#ECE4DB', '#ECE4DB', # index i
            '#0466C8','#0466C8', # play
            '#33415C', '#33415C', # insertion - Suppression
            '#9d4edd', '#9d4edd', # Mirror 1
            '#b76935', '#b76935'] # Mirror 2


colors_exp_2_sober = [
    '#2F4A70', '#2F4A70', # Mirror Rep (deep slate blue)
    '#9FB1CC', '#9FB1CC', # Mirror NoRep (soft muted blue)

    '#4A5D23', '#4A5D23', # play (muted olive green-gray)
    '#A3B18A', '#A3B18A', # play-4 (sage green)

    '#2E2E2E', '#2E2E2E', # sub-programs 1 (earthy brown-orange)
    '#A0A0A0', '#A0A0A0', # sub-programs 2 (warm beige-gold)
]

# Color palet for the figure dl_distance subset_1a (repetition sequences)
blue_colors=[
  '#364B9A','#364B9A', #Rep-2
  '#0f75bd','#0f75bd', #Rep-3
  '#25aae2','#25aae2', #Rep-4
]

# Color palet for the figure dl_distance subset_1b (nested rep sequences)
nested_colors=[
  '#762A83', # Nested
  '#C2A5CF', # Local Rep
  '#C2A5CF', # Global Rep
  '#0f75bd', #Rep-3
]

blue_nest_colors =[
  '#762A83', # Nested
  '#C2A5CF', # Local Rep
  '#C2A5CF', # Global Rep
  '#364B9A','#364B9A', #Rep-2
  '#0f75bd','#0f75bd', #Rep-3
  '#25aae2','#25aae2', #Rep-4
    
  ]

warm_colors=[
  '#6a040f','#6a040f', # Mirror-1
  '#c05761', '#c05761',# Mirror-2
   
  '#faa307', '#faa307', # Play
  '#f48c06', '#f48c06', # Play-4

  '#d00000', '#d00000', # Sub-1
  '#941c2f', '#941c2f', # Sub-2

  '#e9c46a', '#e9c46a', # Index-i
  '#e76f51', '#e76f51', # Insertion / suppression
]
# Legend size for the plot_regression function
legend_size=10
bar_thickness=0.8
bar_frame_width=3 # define linewidht parameter in barh plots

# Fill conditions
#fill_condition_all=np.concatenate(np.tile([])

# ---------------------------------------
# ************Complexity models************
# ---------------------------------------
chunk_comp_array_old=[9.509775,
                  10.754887,
                  8.0,
                  7.754887,
                  6.965784,
                  12.0,
                  5.61470984,
                  9.509775004,
                  5.61470984,
                  9.50977,
                  10.3398,
                  8.32192,
                  8.90689,
                  8.32193,
                  9.321928,
                  8.754887,
                  8.754887,
                  9.0,
                  8.90689,
                  6.90689,
                  6.90689,
                  6.96578,
                  8.491853,
                  8.6438561,
                  9.3219280
                  ]

# Complexity Chunk-Local
chunk_comp_array = [
    12.0,  # play 4 tokens
    12.0,  # control play 4 tokens
    12.0,  # sub-programs 1
    12.0,  # control sub-programs 1
    12.0,  # sub-programs 2
    12.0,  # control sub-programs 2
    9.169925001442312,  # index i
    9.169925001442312,  # control index i
    9.0,  # play
    8.906890595608518,  # control play
    12.0,  # Insertion
    12.0,  # Suppression
    11.169925001442312,  # Mirror-Rep
    11.169925001442312,  # control Mirror-Rep
    12.0,  # Mirror-NoRep
    12.0,  # control Mirror-NoRep
    12.0,  # Repetition-2
    8.906890595608518,  # control Repetition-2
    12.0,  # Repetition-3
    11.584962500721156,  # control Repetition-3
    12.0,  # Repetition-4
    11.584962500721156,  # control Repetition-4
    9.509775004326936,  # Repetition-Nested
    12.0,  # control NoLocal nested
    9.509775004326936,  # control NoGlobal nested
]

# Complexity Chunk-Rep (Global)
chunk_comp_array_global = [
    5.614709844115208,  # play 4 tokens
    5.614709844115208,  # control play 4 tokens
    12.0,  # sub-programs 1
    12.0,  # control sub-programs 1
    12.0,  # sub-programs 2
    12.0,  # control sub-programs 2
    12.0,  # index i
    12.0,  # control index i
    12.0,  # play
    12.0,  # control play
    12.0,  # Insertion
    12.0,  # Suppression
    12.0,  # Mirror-Rep
    12.0,  # control Mirror-Rep
    12.0,  # Mirror-NoRep
    12.0,  # control Mirror-NoRep
    9.509775004326936,  # Repetition-2
    12.0,  # control Repetition-2
    8.0,  # Repetition-3
    12.0,  # control Repetition-3
    6.965784284662087,  # Repetition-4
    12.0,  # control Repetition-4
    5.614709844115208,  # Repetition-Nested
    5.614709844115208,  # control NoLocal nested
    12.0,  # control NoGlobal nested
]


name_complexities=[
    'LoT Complexity',
    'Subjective Complexity',
    'Shannon Entropy',
    'Shannon Entropy Bigram',
    'Lempel-Ziv',
    'Change Complexity',
    'Change Complexity Extended',
    'Algorithmic Complexity',
    'Subsymetries',
    'Chunk Complexity Local',
    'Chunk Complexity Global',
]

aic_values=[
   30011.59,
    30476.97,
    30198.04,
    31153.64,
    30264.89,
    30999.08,
    31316.567039
]

AIC_models={key:value for key,value in zip(name_complexities,aic_values)}



# -- AIC Values for only the 9 sequences of the Base experiment
aic_values_base=[
    7590.058526,
    8378.090948,
    7821.419402,
    8419.572723,
    7985.370794,
    8461.354038,
    8225.430172
]

AIC_models_base={key:value for key,value in zip(name_complexities,aic_values_base)}


# -- AIC Values for only the 9 sequences of the EXTENDED experiment
aic_values_ext=[
    18662.938918,
    18777.861045,
    18927.785273,
    19340.990176,
    18891.144083,
    19080.786168,
    19354.671435
]

# -- AIC values for : experiment 2 AND seq_name_list_exp1_only
aic_values_exp2_rep=[
  6161.755118, #AIC_value_LoT
  6276.880730, #AIC_value_subjective
  6908.272433, #AIC_value_ShannonEntropy
  6623.987869, #AIC_value_ShannonEntropyBigram
  6351.666211, #AIC_value_LempelZiv
  6920.833588, #AIC_value_change
  6525.656380, #AIC_value_algorithmic
  6954.916696, #AIC_value_Subsymetries
  6953.193236, #AIC_value_chunk_local
  6635.228269 # AIC_value_chunk_global
]

# -- BIC (Bayesian Information Criterion) : experiment 2 AND seq_name_list_exp1_only
bic_values_exp2_rep = [
    6183.175126, # BIC_value_LoT
    6298.300737, # BIC_value_subjective
    6929.692441, # BIC_value_ShannonEntropy
    6645.407877, # BIC_value_ShannonEntropyBigram
    6373.086219, # BIC_value_LempelZiv
    6942.253595, # BIC_value_change
    6547.076388, # BIC_value_algorithmic
    6976.336704, # BIC_value_Subsymetries
    6974.613244, # BIC_value_chunk
    6656.648276,
]

aic_values_exp2_rep_corrected=[
  6150.162844, #AIC_value_LoT
  6269.453131, #AIC_value_subjective
  6903.304839, #AIC_value_ShannonEntropy
  6617.435862, #AIC_value_ShannonEntropyBigram
  6342.055998, #AIC_value_LempelZiv
  6910.640888, #AIC_value_change
  6516.256528, #AIC_value_algorithmic
  6943.499622, #AIC_value_Subsymetries
  6945.673370, #AIC_value_chunk_local
  6625.928863 # AIC_value_chunk_global
]

AIC_models_ext={key:value for key,value in zip(name_complexities,aic_values_ext)}

AIC_models_ext_rep={key:value for key,value in zip(name_complexities,aic_values_exp2_rep)}

BIC_models_ext_rep={key:value for key,value in zip(name_complexities,bic_values_exp2_rep)}

# ----------------------------------------------------------------------
# ************** Results subjective complexity experiment **************
# ----------------------------------------------------------------------

mean_subjective_complexity={'Repetition-2': 1.67,
                            'control Repetition-2': 2.93,
                            'Repetition-3': 2.18,
                            'control Repetition-3': 4.03,
                            'Repetition-4': 3.17,
                            'control Repetition-4': 4.83,
                            'Repetition-Nested': 2.35,
                            'control NoLocal nested': 3.49,
                            'control NoGlobal nested': 2.99,
                            'play': 3.38,
                            'control play': 3.72,
                            'play 4 tokens': 4.3,
                            'control play 4 tokens': 4.94,
                            'sub-programs 1': 4.1,
                            'control sub-programs 1': 4.67,
                            'sub-programs 2': 4.69,
                            'control sub-programs 2': 5.53,
                            'Mirror-Rep': 3.85,
                            'control Mirror-Rep': 4.53,
                            'Mirror-NoRep': 4.2,
                            'control Mirror-NoRep': 4.57,
                            'index i': 2.66,
                            'control index i': 2.71,
                            'Suppression': 4.59,
                            'Insertion': 4.01}

mean_revised_subjective_complexity = {'Repetition-2': 1.63,
                            'control Repetition-2': 2.84,
                            'Repetition-3': 2.15,
                            'control Repetition-3': 4.02,
                            'Repetition-4': 3.19,
                            'control Repetition-4': 4.82,
                            'Repetition-Nested': 2.36,
                            'control NoLocal nested': 3.51,
                            'control NoGlobal nested': 2.95,
                            'play': 3.38,
                            'control play': 3.71,
                            'play 4 tokens': 4.26,
                            'control play 4 tokens': 4.89,
                            'sub-programs 1': 4.01,
                            'control sub-programs 1': 4.59,
                            'sub-programs 2': 4.60,
                            'control sub-programs 2': 5.48,
                            'Mirror-Rep': 3.76,
                            'control Mirror-Rep': 4.51,
                            'Mirror-NoRep': 4.2,
                            'control Mirror-NoRep': 4.49,
                            'index i': 2.59,
                            'control index i': 2.71,
                            'Suppression': 4.57,
                            'Insertion': 3.93
}
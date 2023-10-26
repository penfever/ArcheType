from copy import deepcopy
import pandas as pd
import re
from itertools import chain
from pathlib import Path
import numpy as np
import os

try:
  from .const import *
except ImportError:
  from const import *

from fuzzywuzzy import fuzz

import itertools

#MAPPINGS

state_names = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

state_abbreviations = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

country_codes = ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX", "AY", "AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV", "BW", "BX", "BY", "BZ", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH", "CI", "CJ", "CK", "CL", "CM", "CN", "CO", "CP", "CQ", "CR", "CS", "CT", "CU", "CV", "CW", "CX", "CY", "CZ", "DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH", "DI", "DJ", "DK", "DL", "DM", "DN", "DO", "DP", "DQ", "DR", "DS", "DT", "DU", "DV", "DW", "DX", "DY", "DZ", "EA", "EB", "EC", "ED", "EE", "EF", "EG", "EH", "EI", "EJ", "EK", "EL", "EM", "EN", "EO", "EP", "EQ", "ER", "ES", "ET", "EU", "EV", "EW", "EX", "EY", "EZ", "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH", "FI", "FJ", "FK", "FL", "FM", "FN", "FO", "FP", "FQ", "FR", "FS", "FT", "FU", "FV", "FW", "FX", "FY", "FZ", "GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH", "GI", "GJ", "GK", "GL", "GM", "GN", "GO", "GP", "GQ", "GR", "GS", "GT", "GU", "GV", "GW", "GX", "GY", "GZ", "HA", "HB", "HC", "HD", "HE", "HF", "HG", "HH", "HI", "HJ", "HK", "HL", "HM", "HN", "HO", "HP", "HQ", "HR", "HS", "HT", "HU", "HV", "HW", "HX", "HY", "HZ", "IA", "IB", "IC", "ID", "IE", "IF", "IG", "IH", "II", "IJ", "IK", "IL", "IM", "IN", "IO", "IP", "IQ", "IR", "IS", "IT", "IU", "IV", "IW", "IX", "IY", "IZ", "JA", "JB", "JC", "JD", "JE", "JF", "JG", "JH", "JI", "JJ", "JK", "JL", "JM", "JN", "JO", "JP", "JQ", "JR", "JS", "JT", "JU", "JV", "JW", "JX", "JY", "JZ", "KA", "KB", "KC", "KD", "KE", "KF", "KG", "KH", "KI", "KJ", "KK", "KL", "KM", "KN", "KO", "KP", "KQ", "KR", "KS", "KT", "KU", "KV", "KW", "KX", "KY", "KZ", "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LI", "LJ", "LK", "LL", "LM", "LN", "LO", "LP", "LQ", "LR", "LS", "LT", "LU", "LV", "LW", "LX", "LY", "LZ", "MA", "MB", "MC", "MD", "ME", "MF", "MG", "MH", "MI", "MJ", "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA", "NB", "NC", "ND", "NE", "NF", "NG", "NH", "NI", "NJ", "NK", "NL", "NM", "NN", "NO", "NP", "NQ", "NR", "NS", "NT", "NU", "NV", "NW", "NX", "NY", "NZ", "OA", "OB", "OC", "OD", "OE", "OF", "OG", "OH", "OI", "OJ", "OK", "OL", "OM", "ON", "OO", "OP", "OQ", "OR", "OS", "OT", "OU", "OV", "OW", "OX", "OY", "OZ", "PA", "PB", "PC", "PD", "PE", "PF", "PG", "PH", "PI", "PJ", "PK", "PL", "PM", "PN", "PO", "PP", "PQ", "PR", "PS", "PT", "PU", "PV", "PW", "PX", "PY", "PZ", "QA", "QB", "QC", "QD", "QE", "QF", "QG", "QH", "QI", "QJ", "QK", "QL", "QM", "QN", "QO", "QP", "QQ", "QR", "QS", "QT", "QU", "QV", "QW", "QX", "QY", "QZ", "RA", "RB", "RC", "RD", "RE", "RF", "RG", "RH", "RI", "RJ", "RK", "RL", "RM", "RN", "RO", "RP", "RQ", "RR", "RS", "RT", "RU", "RV", "RW", "RX", "RY", "RZ", "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SP", "SQ", "SR", "SS", "ST", "SU", "SV", "SW", "SX", "SY", "SZ", "TA", "TB", "TC", "TD", "TE", "TF", "TG", "TH", "TI", "TJ", "TK", "TL", "TM", "TN", "TO", "TP", "TQ", "TR", "TS", "TT", "TU", "TV", "TW", "TX", "TY", "TZ", "UA", "UB", "UC", "UD", "UE", "UF", "UG", "UH", "UI", "UJ", "UK", "UL", "UM", "UN", "UO", "UP", "UQ", "UR", "US", "UT", "UU", "UV", "UW", "UX", "UY", "UZ", "VA", "VB", "VC", "VD", "VE", "VF", "VG", "VH", "VI", "VJ", "VK", "VL", "VM", "VN", "VO", "VP", "VQ", "VR", "VS", "VT", "VU", "VV", "VW", "VX", "VY", "VZ", "WA", "WB", "WC", "WD", "WE", "WF", "WG", "WH", "WI", "WJ", "WK", "WL", "WM", "WN", "WO", "WP", "WQ", "WR", "WS", "WT", "WU", "WV", "WW", "WX", "WY", "WZ", "XA", "XB", "XC", "XD", "XE", "XF", "XG", "XH", "XI", "XJ", "XK", "XL", "XM", "XN", "XO", "XP", "XQ", "XR", "XS", "XT", "XU", "XV", "XW", "XX", "XY", "XZ", "YA", "YB", "YC", "YD", "YE", "YF", "YG", "YH", "YI", "YJ", "YK", "YL", "YM", "YN", "YO", "YP", "YQ", "YR", "YS", "YT", "YU", "YV", "YW", "YX", "YY", "YZ", "ZA", "ZB", "ZC", "ZD", "ZE", "ZF", "ZG", "ZH", "ZI", "ZJ", "ZK", "ZL", "ZM", "ZN", "ZO", "ZP", "ZQ", "ZR", "ZS", "ZT", "ZU", "ZV", "ZW", "ZX", "ZY", "ZZ"]


label_dict_map = {'Location' : 'streetAddress', 'PostalAddress' : 'streetAddress', 'CreativeWorkSeries' : 'CreativeWork', 'DateTime' : 'Date', 'QuantitativeValue' : "Number", "Integer" : "Number", 
                  "faxNumber" : "telephone", "Email" : "email", "unitText" : "Text", "Mass" : "weight", "MusicRecording" : "MusicAlbum",
                  "MonetaryAmount" : "price", "ProductModel" : "Product", "CoordinateAT" : "Coordinates", 'OccupationalExperienceRequirements' : 'JobRequirements',
                  'Thing' : 'Text', "MusicArtistAT" : "MusicGroup", 'Action' : "WebHTMLAction", "Energy" : "Calories", 'postalCode' : 'zipCode', "LocalBusiness" : "Company",
                  "addressLocality" : "streetAddress", "addressRegion" : "Country", "Place" : "Organization", "WarrantyPromise" : "Text", "typicalAgeRange" : "Age",
                  "EducationalOccupationalCredential" : "JobRequirements", "EventStatusType" : "Event", "identifierNameAP" : "IdentifierAT", "ItemAvailability" : "category", "MusicGroup" : "Artist",
                  "SportsEvent" : "Event"}

label_dict_map_full = {
    'DateTime' : 'calendarvalue',
    'EventStatusType' : 'statustype',
    'EventAttendanceModeEnumeration' : 'attendenum',
    'priceRange' : 'costrange',
    'ItemAvailability' : 'availabilityofitem',
    "LocalBusiness" : "company",
    'addressRegion' : 'countyorstate',
    'addressLocality' : 'city',
    'SportsTeam' : 'athleticteam',
    'ProductModel' : 'modelobject',
    'BookFormatType' : 'formatofbook',
    'CreativeWorkSeries' : 'seriescreative',
    "Energy" : "calories",
    'Action' : "webhtmlaction",
    'Photograph' : 'photourl',
    'ProductModel' : 'modelnameorid',
    'QuantitativeValue' : 'quantityrange',
    'Place' : 'buildingname',
}

cll = ['currency',
 'Product/name',
 'price',
 'DateTime',
 'Date',
 'Number',
 'Integer',
 'Hotel/name',
 'Brand',
 'Text',
 'IdentifierAT',
 'ItemList',
 'Recipe/name',
 'QuantitativeValue',
 'Event/name',
 'Duration',
 'telephone',
 'EventStatusType',
 'PostalAddress',
 'Place',
 'EventAttendanceModeEnumeration',
 'Organization',
 'priceRange',
 'Country',
 'Person',
 'OfferItemCondition',
 'ItemAvailability',
 'email',
 'LocalBusiness/name',
 'postalCode',
 'streetAddress',
 'addressRegion',
 'addressLocality',
 'Person/name',
 'category',
 'URL',
 'MonetaryAmount',
 'Review',
 'Rating',
 'Language',
 'Thing',
 'OccupationalExperienceRequirements',
 'Place/name',
 'CoordinateAT',
 'faxNumber',
 'Mass',
 'CategoryCode',
 'openingHours',
 'identifierNameAP',
 'weight',
 'SportsEvent/name',
 'SportsTeam',
 'ProductModel',
 'Movie/name',
 'CreativeWork/name',
 'unitCode',
 'DeliveryMethod',
 'JobPosting/name',
 'Boolean',
 'Museum/name',
 'Book/name',
 'BookFormatType',
 'TVEpisode/name',
 'CreativeWorkSeries',
 'Energy',
 'RestrictedDiet',
 'Restaurant/name',
 'Action',
 'Photograph',
 'LocationFeatureSpecification',
 'MusicArtistAT',
 'MusicAlbum',
 'EducationalOccupationalCredential',
 'Distance',
 'Product',
 'workHours',
 'Time',
 'GenderType',
 'DayOfWeek',
 'MusicAlbum/name',
 'CreativeWork',
 'EducationalOrganization',
 'MusicRecording/name',
 'paymentAccepted',
 'typicalAgeRange',
 'Offer',
 'unitText',
 'MusicRecording',
 'audience',
 'WarrantyPromise',
 'MusicGroup']

clt = ['WebHTMLAction',
 'Book',
 'Boolean',
 'Brand',
 'Coordinates',
 'Country',
 'CreativeWork',
 'Date',
 'DayOfWeek',
 'DeliveryMethod',
 'Distance',
 'Duration',
 'EducationalOrganization',
 'Calories',
 'Event',
 'GenderType',
 'Hotel',
 'IdentifierAT',
 'ItemList',
 'JobPosting',
 'Language',
 'Company',
 'Movie',
 'Museum',
 'MusicAlbum',
 'Artist',
 'Number',
 'JobRequirements',
 'Offer',
 'Organization',
 'Person',
 'Location',
 'PostalAddress',
 'Product',
 'Rating',
 'Recipe',
 'Restaurant',
 'Review',
 'SportsTeam',
 'TVEpisode',
 'Text',
 'Time',
 'URL',
 'category',
 'currency',
 'email',
 'paymentAccepted',
 'price',
 'streetAddress',
 'telephone',
 'Age',
 'weight',
 'zipCode']

cls = [
 'Boolean',
 'Coordinates',
 'Country',
 'CreativeWork',
 'Date',
 'Event',
 'Gender',
 'JobPosting',
 'Language',
 'Company',
 'Number',
 'Organization',
 'Person',
 'Product',
 'SportsTeam',
 'Text',
 'Time',
 #'URL',
 'category',
 'currency',
 'email',
 'price',
 'streetAddress',
 'telephone',
 'Age',
 'weight',
 'zipCode']

abbrev_map = {**{s[1:] : s for s in cll}, **{s[1:] : s for s in clt}, **{s[2:] : s for s in cll if len(s) > 5}, **{s[2:] : s for s in clt if len(s) > 5}}

context_labels = {"name" : "context_labels", "label_set" : cll, "dict_map" : label_dict_map_full, 'abbrev_map' : abbrev_map}

context_labels_trim = {"name" : "context_labels_trim", "label_set" : clt, "dict_map" : label_dict_map, 'abbrev_map' : abbrev_map}

numeric_labels = ['currency', 'price', 'Number',
       'Integer', 'IdentifierAT', 'QuantitativeValue',
       'Duration',
       'priceRange', 'postalCode', 'MonetaryAmount', 'Mass', 'CategoryCode',
        'weight',
       'unitCode', 'Energy', 'Distance',
       'workHours', 'typicalAgeRange', 'telephone', 'faxNumber', 'zipCode', 'DateTime', 'Date', 'Time', 'PostalAddress', 'openingHours']

always_numeric_labels = ['Number', 'Integer', 'IdentifierAT', 'QuantitativeValue', 'unitCode']

label_dict_map_small = {'Location' : 'streetAddress', 'PostalAddress' : 'streetAddress', 
                        'CreativeWorkSeries' : 'CreativeWork', 'Book' : 'CreativeWork', 'DateTime' : 'Date', 
                        'QuantitativeValue' : "Number", "Integer" : "Number", "GenderType" : "Gender", "IdentifierAT" : "Number",
                        "Hotel" : "Company", "ItemList" : "category", "Movie" : "CreativeWork", "Museum" : "Organization",
                        "JobRequirements" : "category", "calories" : "Number", "Offer" : "Text", "Location" : "streetAddress", "PostalAddress" : "streetAddress",
                        "Rating" : "category", "Restaurant" : "Company", "Review" : "Text", "Recipe" : "Text",
                        "TVEpisode" : "CreativeWork", "paymentAccepted" : "category",
                        "faxNumber" : "telephone", "Email" : "email", "unitText" : "Text", "Mass" : "weight", 
                        "MusicRecording" : "MusicAlbum", "Brand" : "Product", "DayOfWeek" : "Date", "DeliveryMethod" : "Text",
                        "Distance" : "Number", "Duration" : "Time", "EducationalOrganization" : "Organization",
                        "MonetaryAmount" : "price", "ProductModel" : "Product", "CoordinateAT" : "Coordinates", 
                        'OccupationalExperienceRequirements' : 'JobRequirements',
                        'Thing' : 'Text', "MusicArtistAT" : "Person", 'Action' : "URL", 
                        "Energy" : "Number", 'postalCode' : 'zipCode', "LocalBusiness" : "Company",
                        "addressLocality" : "streetAddress", "addressRegion" : "Country", 
                        "Place" : "Organization", "WarrantyPromise" : "Text", "typicalAgeRange" : "Age",
                        "EducationalOccupationalCredential" : "JobRequirements", "EventStatusType" : "Event", 
                        "identifierNameAP" : "Company", "ItemAvailability" : "category", "MusicGroup" : "Person",
                        "SportsEvent" : "Event", "Audience" : "Person", "Energy" : "Number", "EventAttendanceModeEnumeration" : "Boolean", "url" : "URL",
                        "OfferItemCondition" : "category", "MusicAlbum" : "creativework", "MusicRecording" : "creativework",
                        "openingHours" : "Time", "Photograph" : "URL", "priceRange" : "price", "unitCode" : "category", "workHours" : "Time", "CategoryCode" : "category",
                        "RestrictedDiet" : "category", "BookFormatType" : "category", "LocationFeatureSpecification" : "text", "audience" : "category"}

context_labels_small = {"name" : "context_labels_small", "label_set" : cls, "dict_map" : label_dict_map_small, 'abbrev_map' : abbrev_map}

sotab_integer_labels = ['DateTime', 'Date', 'Integer', 'telephone', 'faxNumber', 'Energy']

sotab_float_labels = ['price',
 'Number',
 'QuantitativeValue',
 'Duration',
 'priceRange',
 'MonetaryAmount',
 'CoordinateAT',
 'Mass',
 'weight',
 'Distance']

sotab_other_labels = ['Identifier', 'email', 'URL', 'WebHTMLAction', 'Photograph', 'category', 'text']

sotab_top_hier = {"integer" : sotab_integer_labels, "float" : sotab_float_labels, "other" : sotab_other_labels}

sotab_identifier = ['IdentifierAT', 'CategoryCode', 'identifierNameAP', 'unitCode']

sotab_category = ['currency',
 'ItemList',
 'EventStatusType',
 'EventAttendanceModeEnumeration',
 'OfferItemCondition',
 'ItemAvailability',
 'category',
 'Rating',
 'Language',
 'OccupationalExperienceRequirements',
 'DeliveryMethod',
 'BookFormatType',
 'EducationalOccupationalCredential',
 'GenderType',
 'paymentAccepted',
 'typicalAgeRange']

sotab_text = ['Product/name',
 'Hotel/name',
 'Brand',
 'Text',
 'Recipe/name',
 'Event/name',
 'PostalAddress',
 'Place',
 'Organization',
 'Country',
 'Person',
 'LocalBusiness/name',
 'streetAddress',
 'addressRegion',
 'addressLocality',
 'Person/name',
 'Review',
 'Thing',
 'Place/name',
 'openingHours',
 'SportsEvent/name',
 'SportsTeam',
 'ProductModel',
 'Movie/name',
 'CreativeWork/name',
 'JobPosting/name',
 'Museum/name',
 'Book/name',
 'TVEpisode/name',
 'CreativeWorkSeries',
 'RestrictedDiet',
 'Restaurant/name',
 'LocationFeatureSpecification',
 'MusicArtistAT',
 'MusicAlbum',
 'Product',
 'workHours',
 'Time',
 'DayOfWeek',
 'MusicAlbum/name',
 'CreativeWork',
 'EducationalOrganization',
 'MusicRecording/name',
 'Offer',
 'unitText',
 'MusicRecording',
 'audience',
 'WarrantyPromise',
 'MusicGroup']

sotab_other_hier = {"Identifier" : sotab_identifier, "category" : sotab_category, "text" : sotab_text}

#FUNCTIONS

def fix_labels(label, label_set):
  label = label.lower().strip()
  ldm = {k.lower().strip() : v.lower().strip() for k, v in label_set['dict_map'].items()}
  if label_set.get("abbrev_map", -1) != -1:
    lda = {k.lower().strip() : v.lower().strip() for k, v in label_set['abbrev_map'].items()}
    ldares = lda.get(label, "")
    if ldares != "":
      label = ldares
  if label.endswith("/name"):
    label = label[:-5]
  remap = ldm.get(label, -1)
  if remap != -1:
    label = remap
  return label.lower()

def make_json(prompt, var_params, args):
  p = args["params"]
  if var_params:
    for k, v in var_params.items():
      p[k] = v
  return {
      "data": [
              prompt,
              p['max_new_tokens'],
              p['do_sample'],
              p['temperature'],
              p['top_p'],
              p['typical_p'],
              p['repetition_penalty'],
              p['encoder_repetition_penalty'],
              p['top_k'],
              p['min_length'],
              p['no_repeat_ngram_size'],
              p['num_beams'],
              p['penalty_alpha'],
              p['length_penalty'],
              p['early_stopping'],
              p['seed'],     
          ]
      }

def prompt_context_insert(context_labels: str, context : str, max_len : int = 2000, model : str = "gpt-3.5", args : dict = dict()):
  if "gpt" in model:
    addl_instr = "Please respond only with the name of the class."
  else:
    addl_instr = ""
  if args.get("table_name", -1) != -1:
    table_src_str = " sourced from the table named " + args["table_name"]
  else:
    table_src_str = ""
  if args.get("other_context_samples", -1) != -1:
    ocs = "For additional context, here are some entries from other columns in the table: " + ", ".join(args["other_context_samples"]) + "\n"
  else:
    ocs = ""
  if context_labels == "2step":
    context_labels = "massachusetts, pennsylvania, connecticut, mississippi, washington, california, minnesota, louisiana, tennessee, wisconsin, nebraska, missouri, michigan, kentucky, arkansas, delaware, illinois, colorado, virginia, oklahoma, maryland, indiana, alabama, arizona, georgia, montana, florida, nevada, kansas, alaska, oregon, hawaii, maine, texas, idaho, iowa, ohio"
    s = f'Select the state the articles in the following column are from from {context_labels}. \n Input column: {context} \n{ocs} Output: \n'
  if "chorusprompt" in model:
    s = f'For the following table column{table_src_str}, select a schema.org type annotation from {context_labels}. {addl_instr} \n Input column: {context} \n{ocs} Output: \n'
  elif "koriniprompt" in model:
    s = f'Answer the question based on the task below. If the question cannot be answered using the information provided, answer with "I don\'t know". \n Task: Classify the column{table_src_str} given to you into only one of these types: {context_labels} \n {addl_instr} \n Input column: {context} \n{ocs} Type: \n'
  elif "invertedprompt" in model:
    s = f'Here is a column from a table{table_src_str}: {context} \n{ocs} Please select the class from that best describes the column, from the following options. \n Options: {context_labels} \n {addl_instr} \n Response: \n'
  elif "shortprompt" in model:
    s = f'Pick the column\'s class{table_src_str}. {addl_instr} \n Column: {context} \n{ocs} Classes: {context_labels} \n Output: \n'
  elif "noisyprompt" in model:
    s = f'Pick the column\'s class{table_src_str}. I mean if you want to. It would be cool, I think. Anyway, give it a try, I guess? \n Here\'s the column itself! {context} \n{ocs} And, um, here are some column names you could pick from ... {context_labels} \n {addl_instr} \n Ok, go ahead! \n'
  elif "fozzieprompt" in model:
    s = f'Waka waka! This is Fozzie bear! I would totally ❤️ you if you would be my friend, and also pick a class for this column{table_src_str}, before we end. \n Here\'s the column, waka waka! {context} \n{ocs} If you get the right class, it\'ll be a real gas! {context_labels} \n {addl_instr} \n What\'s the type? \n'
  elif "gpt" in model:
    s = f'SYSTEM: Please select the class from {context_labels} which best describes the context{table_src_str}. {addl_instr} \n CONTEXT: {context} \n{ocs} \n RESPONSE: \n'
  elif model == "llama-old":
    s = f'INSTRUCTION: Select the class{table_src_str} from the category which matches the input. \n CATEGORIES: {context_labels} \n INPUT:{context} \n{ocs} {addl_instr} \n OUTPUT: '
  elif "-zs" in model:
    ct = "[" + ", ".join(context).replace("[", "").replace("]", "").replace("'", "") + "]"
    lb = "\n".join(["- " + c for c in context_labels.split(", ")])
    if model == "opt-iml-max-30b-zs":
        s = f'Select the option{table_src_str} which best describes the input. \n INPUT: {ct} .\n{ocs} {addl_instr} \n OPTIONS:\n {lb} \n'
    else:
        s = f'INSTRUCTION: Select the option{table_src_str} which best describes the input. \n INPUT: {ct} .\n{ocs} {addl_instr} \n OPTIONS:\n {lb} \n ANSWER: '
  elif model in ["llama", "ArcheType-llama", "ArcheType-llama-oc"]:
    s = f'INSTRUCTION: Select the category{table_src_str} which best matches the input. \n INPUT:{context} \n{ocs} {addl_instr} \n CATEGORY: '
  if "internlm" in model or "speechless" in model:
    s = s.replace("[", "").replace("]", "").replace("'", "")
  if args.get('tokenizer', None) is not None and len(s) > max_len:
    inputs = args["tokenizer"].encode(s, return_tensors="pt")
    target_len = len(inputs[0])
    if target_len > max_len:
      inputs = inputs[:,:max_len-len(context_labels)-100]
      s = args["tokenizer"].decode(inputs[0]) + f"Classes: {context_labels} \n Output: \n"
  return s

def prompt_2step_context_insert(context : str, max_len : int = 2000, model : str = "gpt-3.5", args : dict = dict()):
  contexr_labels = "massachusetts, pennsylvania, connecticut, mississippi, washington, california, minnesota, louisiana, tennessee, wisconsin, nebraska, missouri, michigan, kentucky, arkansas, delaware, illinois, colorado, virginia, oklahoma, maryland, indiana, alabama, arizona, georgia, montana, florida, nevada, kansas, alaska, oregon, hawaii, maine, texas, idaho, iowa, ohio"

def derive_meta_features(col):
  features = {}
  if not col.astype(str).apply(str.isnumeric).all():
    return {"std" : round(col.astype(str).str.len().std(), 2), "mean" : round(col.astype(str).str.len().mean(), 2), "mode" : col.astype(str).str.len().mode().iloc[0].item(), "median" : col.astype(str).str.len().median(), "max" : col.astype(str).str.len().max(), "min" : col.astype(str).str.len().min(), "rolling-mean-window-4" : [0.0]}
  col = col.dropna().astype(float)
  if col.apply(float.is_integer).all():
    col = col.astype(int)
  features['std'] = round(col.std(), 2)
  features['mean'] = round(col.mean(), 2)
  features['mode'] = col.mode().iloc[0].item()
  features['median'] = col.median()
  features['max'] = col.max()
  features['min'] = col.min()
  indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
  features['rolling-mean-window-4'] = list(col.rolling(window=indexer, min_periods=1).mean())
  return features

def insert_source(context, fname, zs=False):
  pattern = r"_([^_]*)_" # Matches substrings that start and end with "_"
  matcher = re.search(pattern, fname)
  addstr = str(matcher.group()).replace("_", "").split(".")[0]
  if zs:
    return addstr
  else:
    context.insert(0, "SRC: " + addstr)
    return context    

def get_all_substrings(input_list, chars=None):
    """
    Given a list of strings, returns a flat list of all substrings split on " ".
    
    Parameters:
        input_list (list of str): A list of strings
    
    Returns:
        list of str: A flattened list of all substrings.
    """
    # Split each string in input_list and chain them to create a flat list
    if isinstance(input_list, str):
      input_list = [input_list]
    if chars:
      for c in chars:
        input_list = [s.replace(c, " ") for s in input_list]
    return list(itertools.chain.from_iterable([s.split(" ") for s in input_list]))

def fuzzy_substring(sub, s, threshold=85):
    """
    Check if a substring approximately exists in a string.
    
    Parameters:
    - sub (str): The substring to find.
    - s (str): The string to search within.
    - threshold (int): The minimum similarity ratio.

    Returns:
    - bool: True if there's a fuzzy match, False otherwise.
    """
    # Make the search case-insensitive
    sub = sub.lower()
    s = s.lower()

    threshold_max = 0

    surrounding_words = ""

    # We'll search every contiguous substring of length len(sub) in s
    for i in range(len(s) - len(sub) + 1):
        new_thresh = fuzz.ratio(sub, s[i:i+len(sub)])
        if new_thresh >= threshold_max:
            threshold_max = new_thresh
            start_pos = max(i - 300, 0)
            end_pos = min(i + len(sub) + 300, len(s))
            surrounding_words = s[start_pos:end_pos]
    return threshold_max, surrounding_words

def get_df_sample(df, rand_seed, val_indices, len_context, min_variance=1, replace=False, full=False, other_col=False, max_len=8000, method=[], coherence_scores=None, args=dict()):
    column_samples = {}
    ignore_list = ["None", 'none', 'NaN', 'nan', 'N/A', 'na', '']
    for idx, col in enumerate(df.columns):
      colvals = df.astype(str)[col]
      ss_orig = max_len//(len_context*3)
      if "simple_random_sampling" in method:
        sample_list = colvals.sample(n=ss_orig, replace=True, random_state=rand_seed).tolist()
      elif "first_sampling" in method:
        colvalsl = colvals.tolist()
        sample_list = colvalsl[:ss_orig]
      elif "coherence_sampling" in method:
        sample_list = colvals.sample(n=ss_orig, replace=True, random_state=rand_seed, weights=coherence_scores[idx]).tolist()
      else:
        #archetype sampling
        sample_list = []
        for p in sorted(pd.unique(colvals).tolist()):
          if p in sample_list or p in ignore_list:
            continue
          sample_list.append(p)
        #sort the list twice, first alphabetically, then by length (in case of ties)
        sample_list = sorted(sample_list, key=len, reverse=True)
        if "amstr_weighted_sampling" in method:
          if all([len(s) < 600 for s in sample_list]):
            weights = np.linspace(1, 0.1, len(sample_list))
          else:
            weights = np.zeros(len(sample_list))
            for i, s in enumerate(sample_list):
              s = s.replace("<unk>", " ")
              max_thresh = 0
              ret_words = ""
              for k in state_names:
                thresh, words = fuzzy_substring(k, s)
                if thresh > max_thresh:
                  max_thresh = thresh
                  ret_words = words
              s = ret_words
              weights[i] = max_thresh
        else:
          weights = np.linspace(1, 0.1, len(sample_list))
        weights = weights / np.sum(weights)
        if len(sample_list) > ss_orig:
          sample_list = np.array(sample_list)
          np.random.seed(rand_seed)
          indices = np.random.choice(np.arange(len(sample_list)), size=ss_orig, replace=replace, p=weights)
          sample_list = sample_list[indices].tolist()
      #reformat integer samples
      sl_mod = []
      # Meta-features
      if full:
        meta_features = derive_meta_features(df[col])
        meta_features['rolling-mean-window-4'] = meta_features['rolling-mean-window-4'][:5]
      # Sampling from other columns
      if other_col:
        sample_list_fill_size = len_context - len(sample_list)
        nc = len(df.columns)
        per_column_context = max(1, sample_list_fill_size // nc)
        for idx, oc in enumerate(df.columns):
          items = df[oc].astype(str).iloc[0:per_column_context].tolist()
          model_name = args.get("model_name", "")
          if "zs" not in model_name:
            sample_list = sample_list + [f"OC_{idx}: " + str(item) for item in items]
          else:
            if args.get("other_context_samples", -1) == -1:
              args["other_context_samples"] = []
            args["other_context_samples"] = args["other_context_samples"] + [str(item) for item in items]
      if not sample_list:
        sample_list = ["None"]
      if len(sample_list) < len_context:
        sample_list = sample_list * len_context
      if len(sample_list) > len_context:
        sample_list = sample_list[:len_context]
      assert len(sample_list) == len_context, "An index in val_indices is length " + str(len(sample_list))
      if full:
        if meta_features['std'] == "N/A":
          sample_list = sample_list + ["" for k,v in meta_features.items()]
        else:
          sample_list = sample_list + [str(k) + ": " + str(v) for k,v in meta_features.items()]
      column_samples[col] = sample_list
    return pd.DataFrame.from_dict(column_samples)

#SHERLOCK
# Get the current script directory path
current_script_dir = ARCHETYPE_PATH

# Define the relative path to file.csv from the current script directory
relative_path_to_csv = os.path.join(current_script_dir, 'metadata', 'wotab-mapping.csv')

# Normalize the path to handle any inconsistencies in the directory separators
csv_path = os.path.normpath(relative_path_to_csv)

mappings = pd.read_csv(csv_path)

#SCHEMA ORG
relative_path_to_csv = os.path.join(current_script_dir, 'metadata', 'schemaorg-current-https-types.csv')

# Normalize the path to handle any inconsistencies in the directory separators
csv_path = os.path.normpath(relative_path_to_csv)

schema_df = pd.read_csv(csv_path)

def get_schema_df():
  return schema_df


sherlock_to_cta = {}
cta_list = list(set(mappings['Sherlock CTA'].tolist()))
for mapping in cta_list:
  mapping_split = mapping.split(", ")
  for m in mapping_split:
    if not m:
      continue
    map_list = list(set(mappings[mappings['Sherlock CTA'].str.contains(m)]['CTA label'].tolist()))
    match_set = list(chain(*[k.split(", ") for k in map_list]))
    match_set = list(set([fix_labels(m, context_labels_trim) for m in match_set]))
    if not match_set:
      match_set = ["NoMatch"]
    m = m.lower()
    if sherlock_to_cta.get(m, -1) != -1:
      sherlock_to_cta[m] = list(set(sherlock_to_cta[m] + match_set))
    else:
      sherlock_to_cta[m] = match_set

sherlock_labels = ["Address", "Affiliate", "Affiliation", "Age", "Album", "Area", "Artist", "Birth date", "Birth place", "Brand", "Capacity", "Category", "City", "Class", "Classification", "Club", "Code", "Collection ", "Command ", "Company", "Component", "Continent", "Country", "County", "Creator", "Credit", "Currency", "Day", "Depth", "Description", "Director", "Duration", "Education", "Elevation ", "Family ", "File size", "Format", "Gender", "Genre", "Grades", "ISBN", "Industry", "Jockey", "Language", "Location", "Manufacturer", "Name", "Nationality", "Notes", "Operator", "Order", "Organisation", "Origin", "Owner", "Person", "Plays", "Position", "Product", "Publisher", "Range", "Rank", "Ranking", "Region", "Religion", "Requirement", "Result", "Sales", "Service", "Sex", "Species", "State", "Status", "Symbol", "Team", "Team name", "Type", "Weight", "Year"]
sherlock_map_reverse = {s.lower() : i for i, s in enumerate(sherlock_labels)}
sherlock_map_forward = {i : s.lower() for i, s in enumerate(sherlock_labels)}

#D4

# Define the relative path to file.csv from the current script directory
relative_path_to_d4 = os.path.join(current_script_dir, 'metadata', 'D4')
# Normalize the path to handle any inconsistencies in the directory separators
d4p = os.path.normpath(relative_path_to_d4)
D4_PATH = Path(d4p)

def get_d4_dfs():
  D4_files = sorted(list(D4_PATH.rglob("**/*.silver")))
  NUM_SAMPLES = 400
  d4_dfs = {}
  for f in D4_files:
      df = pd.read_csv(f, sep='\t', names=["ID1", "ID2", "values"])
      for k in range(NUM_SAMPLES):
          dfs = df.sample(n=1 + np.random.randint(100), replace=True, random_state=k)
          d4_dfs[str(f.stem).lower() + "_" + str(k)] = dfs
  return d4_dfs

D4_renamed_classes = ['school-number',
 'Ethnicity',
 'school-grades',
 'School Name',
 'school-dbn',
 'Elevator or Staircase',
 'Region in Brooklyn',
 'Region in Bronx',
 'Region in Queens',
 'Region in Manhattan',
 'Region in Staten Island',
 'permit-types',
 'borough',
 'Abbreviation of Agency',
 'Color',
 'NYC Agency Name',
 'other-states',
 'us-state',
 'Month',
 'plate-type']

D4_classes = ['school-number', 
              'ethnicity', 
              'school-grades', 
              'school-name', 
              'school-dbn', 
              'rental-building-class', 
              'brooklyn', 
              'bronx', 
              'queens', 
              'manhattan', 
              'staten_island', 
              'permit-types',
              'borough',  
              'agency-short', 
              'color', 
              'agency-full', 
              'other-states', 
              'us-state', 
              'month',
              'plate-type']

D4_classname_map = {k1 : k2 for (k1, k2) in zip(D4_classes, D4_renamed_classes)}

sherlock_D4_map = {
  'school-number' : ['Code'],
  'ethnicity' : ['Nationality'],
  'school-grades' : ['Description', 'Grades'],
  'school-name' : ['Organisation', 'Education'],
  'school-dbn' : ['Description', 'Code'],
  'brooklyn' : ['Region', 'Address', 'Location'],
  'bronx' : ['Region', 'Address', 'Location'],
  'queens' : ['Region', 'Address', 'Location'],
  'manhattan' : ['Region', 'Address', 'Location'],
  'staten_island' : ['Region', 'Address', 'Location'],
  'borough' : ['County'],
  'color' : ['Description', 'Type', 'Category'],
  'permit-types' : ['Symbol', 'Code'],
  'rental-building-class' : ['Requirement', 'Operator'],
  'agency-short' : ['Organisation', 'Affiliate'],
  'agency-full' : ['Organisation', 'Affiliate'],
  'other-states' : ['Country'],
  'us-state' : ['State'],
  'month' : ['Birth date', 'Day'],
  'plate-type' : ['Symbol', 'Type'],
}

sotab_D4_map = {
  'school-number' : ['identifierNameAP', 'IdentifierAT', 'Text', 'Number', 'Integer', 'QuantitativeValue'],
  'ethnicity' : ['Person', 'Person/name', 'category'],
  'school-grades' : ['Text', 'category', 'CategoryCode', 'Offer'],
  'school-name' : ['Organization', 'EducationalOrganization'],
  'school-dbn' : ['Text', 'identifierNameAP', 'IdentifierAT', 'unitCode'],
  'brooklyn' : ['streetAddress', 'addressRegion', 'Place', 'addressLocality'],
  'bronx' : ['streetAddress', 'addressRegion', 'Place', 'addressLocality'],
  'queens' : ['streetAddress', 'addressRegion', 'Place', 'addressLocality'],
  'manhattan' : ['streetAddress', 'addressRegion', 'Place', 'addressLocality'],
  'staten_island' : ['streetAddress', 'addressRegion', 'Place', 'addressLocality'],
  'borough' : ['addressRegion', 'Place', 'addressLocality'],
  'color' : ['Text', 'category'],
  'permit-types' : ['Text', 'identifierNameAP', 'IdentifierAT', 'unitCode'],
  'rental-building-class' : ['Thing', 'category', "LocationFeatureSpecification"],
  'agency-short' : ['Organization', 'EducationalOrganization'],
  'agency-full' : ['Organization', 'EducationalOrganization'],
  'other-states' : ['Country'],
  'us-state' : ["addressRegion", "addressLocality"],
  'month' : ["DateTime", "Date"],
  'plate-type' : ['Text', 'category', 'CategoryCode', 'Offer'],
}

# d4_zs_context_labels = {"name" : "d4_zs", "label_set" : D4_renamed_classes, "dict_map" : {c : c for c in D4_renamed_classes}, "d4_map" : D4_classname_map}

d4_zs_context_labels = {"name" : "d4_zs", "label_set" : D4_renamed_classes, "dict_map" : D4_classname_map}

d4_sotab_labels = {"name" : "d4_sotab", "label_set" : cll, "dict_map" : {c : c for c in cll}, "d4_map" : sotab_D4_map}

d4_sherlock_labels = {"name" : "d4_sherlock", "label_set" : sherlock_labels, "dict_map" : {c : c for c in sherlock_labels}, "d4_map" : sherlock_D4_map}


# amstr_tables

def get_amstr_dfs(amstr_path, rand_seed):
  amstr_files = sorted(list(Path(amstr_path).rglob("*.csv")))
  amstr_dfs = {}
  for f in amstr_files:
    df = pd.read_csv(f)
    for column_name, column_data in df.items():
      dfs = column_data.sample(15, replace=True, random_state=rand_seed)
      dfs = pd.DataFrame(dfs)
      amstr_dfs[str(column_name) + "_" + str(f.stem).split('_')[-1]] = dfs
  return amstr_dfs

amstr_classes = ['title', 'issn', 'town', 'state', 'headline', 'byline'] + \
                [state.replace(" ", "_") + "_article" for state in state_names] + \
                [state + "_article" for state in state_names]


amstr_renamed_class = ['Newspaper or Publication','Numeric Identifier','Town','State','Headline','Author Byline'] + \
                      ["Article from " + state for state in state_names] + \
                      ["Article from " + state for state in state_names]

amstr_classname_map = {k1 : k2 for (k1, k2) in zip(amstr_classes, amstr_renamed_class)}

def get_amstr_classname_map():
  return amstr_classname_map

amstr_zs_context_labels = {"name" : "amstr_zs", "label_set" : amstr_renamed_class, "dict_map" : {c : c for c in amstr_renamed_class}, "amstr_map" : amstr_classname_map}

amstr_2step_renamed_classes = ['Newspaper or Publication','Numeric Identifier','Town','State','Headline','Author Byline', 'Article']
amstr_zs_2step_context_labels = {"name" : "amstr_zs_2step", "label_set" : amstr_2step_renamed_classes, "dict_map" : {c : c for c in amstr_renamed_class}, "amstr_map" : amstr_classname_map}


# pubchem

def get_pubchem_dfs(pubchem_path, random_seed):
  pubchem_files = sorted(list(Path(pubchem_path).rglob("*.csv")))
  pubchem_dfs = {}
  for f in pubchem_files:
    df = pd.read_csv(f)
    for column_name, column_data in df.items():
      dfs = column_data.sample(15, replace=True, random_state=random_seed)
      dfs = pd.DataFrame(dfs)
      pubchem_dfs[str(column_name) + "_" + str(f.stem).split('_')[-1]] = dfs
  return pubchem_dfs

pubchem_classes = ['author_given_name', 'Molecular_Formula', 'book_title', 'cell_altlabel', 'book_subtitle', 
                   'disease_altlabel', 'book_creator', 'author_family_name', 'IUPAC_name', 'taxonomy_label', 
                   'InChI', 'SMILES', 'patent_abstract', 
                   'organization_fn', 'book_isbn', 'concept_broader', 'journal_title', 'concept_preflabel', 
                   'author_fn', 'journal_issn', 'patent_title']


pubchem_renamed_class = ['Person\'s First Name and Middle Initials', 'Molecular Formula', 'Book Title', 'Cell Alternative Label', 'Book Title',
                        'Disease Alternative Label', 'MD5 Hash', 'Person\'s Last Name', 'Biological Formula', 'Taxonomy Label',
                        'InChI (International Chemical Identifier)', 'SMILES (Simplified Molecular Input Line Entry System)', 'Abstract for Patent', 
                        'Organization', 'Book ISBN', 'Concept Broader Term', 'Journal Title', 'Chemical',
                        'Person\'s Full Name', 'Journal ISSN', 'Patent Title']

pubchem_classname_map = {k1 : k2 for (k1, k2) in zip(pubchem_classes, pubchem_renamed_class)}

def get_pubchem_classname_map():
  return pubchem_classname_map

pubchem_zs_context_labels = {"name" : "pubchem_zs", "label_set" : pubchem_renamed_class, "dict_map" : {c : c for c in pubchem_renamed_class}, "pubchem_map" : pubchem_classname_map}


def get_lsd(s):
  if s == "SOTAB-91":
    return context_labels
  elif s == "SOTAB-55":
    return context_labels_trim
  elif s == "SOTAB-27":
    return context_labels_small
  elif s == "D4-ZS":
    return d4_zs_context_labels
  elif s == "D4-DoDuo":
    return d4_sherlock_labels
  elif s == "amstr-ZS":
    return amstr_zs_context_labels
  elif s == "amstr-ZS-2step":
    return amstr_zs_2step_context_labels
  elif s == "pubchem-ZS":
    return pubchem_zs_context_labels
  print("Label set not found")
  return None

def pd_read_any(file):
    file = str(file)
    if file.endswith('.csv') or file.endswith('.tsv'):
        df = pd.read_csv(file)
    elif file.endswith('.json'):
        df = pd.read_json(file)
    elif file.endswith('.xml'):
        df = pd.read_xml(file)
    elif file.endswith('.xls') or file.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('.hdf'):
        df = pd.read_hdf(file)           
    elif file.endswith('.sql'):
        df = pd.read_sql(file)
    elif file.endswith('.parquet'):
        df = pd.read_parquet(file)
    else:
        raise ValueError(f'Unsupported filetype: {file}')
    return df
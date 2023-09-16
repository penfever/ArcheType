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

#MAPPINGS

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
 'URL',
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
       'workHours', 'typicalAgeRange']

label_dict_map_small = {'Location' : 'streetAddress', 'PostalAddress' : 'streetAddress', 
                        'CreativeWorkSeries' : 'CreativeWork', 'Book' : 'CreativeWork', 'DateTime' : 'Date', 
                        'QuantitativeValue' : "Number", "Integer" : "Number", "GenderType" : "Gender", "IdentifierAT" : "Text",
                        "Hotel" : "Company", "ItemList" : "category", "Movie" : "CreativeWork", "Museum" : "Organization",
                        "JobRequirements" : "category", "Offer" : "Text", "Location" : "streetAddress", "PostalAddress" : "streetAddress",
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
                        "identifierNameAP" : "IdentifierAT", "ItemAvailability" : "category", "MusicGroup" : "Person",
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

def prompt_context_insert(context_labels: str, context : str, max_len : int = 2000, model : str = "gpt-3.5"):
  if "chorusprompt" in model:
    s = f'For the following table column, select a schema.org type annotation from {context_labels}. \n Input column: {context} \n Output: \n'
  elif "koriniprompt" in model:
    s = f'Answer the question based on the task below. If the question cannot be answered using the information provided, answer with "I don\'t know". \n Task: Classify the column given to you into only one of these types: {context_labels} \n Input column: {context} \n  Type: \n'
  elif "invertedprompt" in model:
    s = f'Here is a column from a table: {context} \n Please select the class from that best describes the column, from the following options. \n Options: {context_labels} \n Response: \n'
  elif "shortprompt" in model:
    s = f'Pick the column\'s class. \n Column: {context} \n Classes: {context_labels} \n Output: \n'
  elif "noisyprompt" in model:
    s = f'Pick the column\'s class. I mean if you want to. It would be cool, I think. Anyway, give it a try, I guess? \n Here\'s the column itself! {context} \n And, um, here are some column names you could pick from ... {context_labels} \n Ok, go ahead! \n'
  elif "fozzieprompt" in model:
    s = f'Waka waka! This is Fozzie bear! I would totally ❤️ you if you would be my friend, and also pick a class for this column, before we end. \n Here\'s the column, waka waka! {context} \n If you get the right class, it\'ll be a real gas! {context_labels} \n What\'s the type? \n'
  elif "gpt-3.5" in model:
    s = f'SYSTEM: Please select the field from {context_labels} which best describes the context. Respond only with the name of the field. \n CONTEXT: {context}'
  elif model == "llama-old":
    s = f'INSTRUCTION: Select the field from the category which matches the input. \n CATEGORIES: {context_labels} \n INPUT:{context} \n OUTPUT: '
  elif "-zs" in model:
    ct = "[" + ", ".join(context).replace("[", "").replace("]", "").replace("'", "")[:max_len - 100 - len(context_labels)] + "]"
    lb = "\n".join(["- " + c for c in context_labels.split(", ")])
    #s = f'How might one classify the following input? \n INPUT: {ct} .\n OPTIONS:\n {lb} \n ANSWER:'
    if model == "opt-iml-max-30b-zs":
        s = f'Select the option which best describes the input. \n INPUT: {ct} .\n OPTIONS:\n {lb} \n'
    else:
        s = f'INSTRUCTION: Select the option which best describes the input. \n INPUT: {ct} .\n OPTIONS:\n {lb} \n ANSWER:'
  elif model in ["llama", "ArcheType-llama", "ArcheType-llama-oc"]:
    s = f'INSTRUCTION: Select the category which best matches the input. \n INPUT:{context} \n CATEGORY: '
  if len(s) > max_len:
    s = s[:max_len - 25]
  return s

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

def insert_source(context, fname):
  pattern = r"_([^_]*)_" # Matches substrings that start and end with "_"
  matcher = re.search(pattern, fname)
  addstr = str(matcher.group()).replace("_", "").split(".")[0]
  context.insert(0, "SRC: " + addstr)
  return context    
    
def get_df_sample(df, rand_seed, val_indices, len_context, min_variance=1, replace=False, full=False, other_col=False, max_len=8000, method=[], coherence_scores=None):
    column_samples = {}
    ignore_list = ["None", 'none', 'NaN', 'nan', 'N/A', 'na', '']
    for idx, col in enumerate(df.columns):
      colvals = df.astype(str)[col]
      if "simple_random_sampling" in method:
        sample_list = colvals.sample(n=max_len//(len_context*3), replace=True, random_state=rand_seed).tolist()
      elif "coherence_sampling" in method:
        sample_list = colvals.sample(n=max_len//(len_context*3), replace=True, random_state=rand_seed, weights=coherence_scores[idx]).tolist()
      else:
        sample_list = list(set(p[:max_len//(len_context*3)] for p in pd.unique(colvals) if p not in ignore_list))
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
          sample_list = sample_list + [f"OC_{idx}: " + str(item) for item in items]
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
D4_files = list(D4_PATH.rglob("**/*.silver"))
D4_classes = list(f.stem.lower() for f in D4_files)

def get_d4_dfs():
  NUM_SAMPLES = 100
  d4_dfs = {}
  for file in D4_files:
      df = pd.read_csv(file, sep='\t', names=["ID1", "ID2", "values"])
      for k in range(NUM_SAMPLES):
          dfs = df.sample(n=1 + np.random.randint(100), replace=True, random_state=k)
          d4_dfs[str(file.stem).lower() + "_" + str(k)] = dfs
  return d4_dfs

D4_renamed_classes = ['School ID',
 'Ethnicity',
 'Letter Grade',
 'Educational Organization',
 'School DBN',
 'Region in Brooklyn',
 'Region in Bronx',
 'Permit Type',
 'Region in Queens',
 'Region in Manhattan',
 'Region in Staten Island',
 'County',
 'Elevator or Staircase',
 'Short City Agency Name',
 'Color',
 'Full City Agency Name',
 'Country',
 'State',
 'Month',
 'License plate type']

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

d4_zs_context_labels = {"name" : "d4_zs", "label_set" : D4_renamed_classes, "dict_map" : {c : c for c in D4_renamed_classes}, "d4_map" : D4_classname_map}

d4_sotab_labels = {"name" : "d4_sotab", "label_set" : cll, "dict_map" : {c : c for c in cll}, "d4_map" : sotab_D4_map}

d4_sherlock_labels = {"name" : "d4_sherlock", "label_set" : sherlock_labels, "dict_map" : {c : c for c in sherlock_labels}, "d4_map" : sherlock_D4_map}

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
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

#|%%--%%| <oMPk4TbLxT|RbZHjyRy4x>

import json
import pandas as pd
import numpy as np

coursera_ = pd.read_json('./output_final.json')['modules'].copy()

# Mapping plural to singular to unify the keys
plural_to_singular = {
    'videos': 'video',
    'quizzes': 'quiz',
    'readings': 'reading',
    'assignments': 'assignment',
    'plugins': 'plugin'
}

uniques = set()

for moduluess_ in coursera_:
    if isinstance(moduluess_, list):
        for weeks in moduluess_:
            overview = weeks.get('over_view', {})
            for key in overview:
                # Convert plural to singular
                normalized_key = plural_to_singular.get(key, key)
                uniques.add(normalized_key)

def process_course_data(data, uniques):
    # Ensure description key exists
    description = f"Description: {data.get('description', 'No description')} Modules:\n"
    
    if isinstance(data.get('modules'), list):
        for module in data['modules']:
            description += f"{module.get('title', 'No title')}: {module.get('description', 'No description')}\n\n"
    elif isinstance(data.get('modules'), dict):
        for key, value in data['modules'].items():
            description += f"{key}: {value}\n\n"
    
    data['description'] = description
    
    # Initialize totals with unified keys
    totals = {key: 0 for key in uniques}

    if isinstance(data.get('modules'), list):
        for module in data['modules']:
            overview = module.get('over_view', {})
            for key, value in overview.items():
                normalized_key = plural_to_singular.get(key, key)  # Normalize key
                totals[normalized_key] += int(value)

    # If the course type is "project", set totals to NaN
    if data.get("type") == "project":
        for key in totals.keys():
            data[f'total_{key}'] = np.nan
    else:
        for key, value in totals.items():
            data[f'total_{key}'] = value

    # One-hot encoding for all module components
    for key in totals.keys():
        data[f'has_{key}'] = 1 if pd.notna(data[f'total_{key}']) and (data[f'total_{key}'] != 0 or isinstance(data[f'total_{key}'], (int, float))) else 0

    # Handle missing ratings
    if data.get('rating', '').lower() == "no rating":
        data['nu_reviews'] = 0
    data['has_no_enrol'] = 0
    data['enrollments'] = np.nan
    data['has_rating'] = 1
    data['subject'] = np.nan
    data['has_subject'] = 0
    data['provider'] = 'coursera'
    
    return data
# Load JSON
with open("./output_final.json", "r") as file:
    course_data_list = json.load(file)

# Process multiple courses if applicable
if isinstance(course_data_list, list):
    processed_data_list = [process_course_data(course, uniques) for course in course_data_list]

if isinstance(processed_data_list, list):
    for course in processed_data_list:
        course.pop("modules", None)  # Remove 'modules' if it exists
# Save output
with open("processed_coursera_data.json", "w") as file:
    json.dump(processed_data_list, file, indent=4, ensure_ascii= False)
#|%%--%%| <RbZHjyRy4x|aFOSSXY89O>

import pandas as pd

edx_courses = pd.read_json('./edx_programs.json').copy()

def pre_edx(data):
    data['description'] = data[['primary_description', 'secondary_description', 'tertiary_description']].fillna('').agg(' '.join, axis=1)
    
    # Extract only 'skill' values from the list of dictionaries in 'skills'
    data['skills'] = data['skills'].apply(lambda x: [skill['skill'] for skill in x if 'skill' in skill] if isinstance(x, list) else [])
    data['rating'] = np.nan
    data['nu_reviews'] = np.nan
    data['has_rating'] = 0
    data['Duration'] = data['weeks_to_complete'] * ((data['max_effort'] + data['min_effort'])/2)
    data['reviews'] = np.nan
    listy = ['_quiz' ,'_plugin' ,'_programming' ,'_peer' ,'_reading' ,'_discussion' ,'_video' ,'_ungraded', '_app', '_assignment', '_teammate']
    for i in listy:
        data[f"total{i}"] = np.nan
        data[f'has{i}'] = 0
    data['has_no_enrol'] = 1
    data['has_subject'] = 1
    data["description"] = data.apply(
        lambda row: row["description"] + ", " + ", ".join(row["course_titles"])
        if row["course_titles"] else row["description"], axis=1
    )

    def replace_introductory(level):
        if isinstance(level, list):
            return ["Beginner" if item == "Introductory" else item for item in level]
        return "Beginner" if level == "Introductory" else level
    data['level'] = data['level'].apply(replace_introductory)
    #preprocessing need on the part of description to add modules 
    # Select and rename columns
    data.rename(columns={
        'title': 'course_name',
        'recent_enrollment_count': 'enrollments',
        'partner': 'organization',
        'marketing_url': 'url',
        'product': 'type',
        'course_titles':'modules',
        'staff': 'instructor'
    }, inplace= True)
    data['provider'] = 'edx'
    lisstyyy = ['url', 'type', 'course_name', 'organization', 'instructor', 'rating', 'nu_reviews', 'description', 'skills', 'level', 'Duration', 'reviews'] + [f"total{i}" for i in listy] + [f"has{j}" for j in listy] + ["has_no_enrol", 'has_rating', "subject", 'has_subject', 'enrollments', 'provider']

    return data[lisstyyy]
   #]]
edx_stuff = pre_edx(edx_courses)



#|%%--%%| <aFOSSXY89O|uBpi2Am1Sr>

print(processed_data_list.columns[processed_data_list.columns.duplicated()])
print(edx_stuff.columns[edx_stuff.columns.duplicated()])

#|%%--%%| <uBpi2Am1Sr|Zk5xEFN0YK>


# Ensure both DataFrames have exactly the same columns
processed_data_list = pd.read_json('./processed_coursera_data.json')
if set(processed_data_list.columns) == set(edx_stuff.columns):
    final_dataset = pd.concat([processed_data_list, edx_stuff], ignore_index=True)
    final_dataset.to_json("combined_dataset.json", orient="records", indent=4, force_ascii= False)
    print("Successfully merged Coursera and edX datasets!")
else:
    missing_in_coursera = set(edx_stuff.columns) - set(processed_data_list.columns)
    missing_in_edx = set(processed_data_list.columns) - set(edx_stuff.columns)
    
    raise ValueError(f"Columns mismatch!\n"
                     f"Missing in Coursera: {missing_in_coursera}\n"
                     f"Missing in edX: {missing_in_edx}")

#|%%--%%| <Zk5xEFN0YK|lE0uO6ro20>

import pandas as pd

data= pd.read_json('../udemy/courses/courses_scrape/courses_scrape/spiders/output.json')
karar = pd.DataFrame()
def title(dataa):
    return dataa['add_to_cart'][0]['title']
karar['title']=data.apply(title, axis = 1)
data_cleaned = data.drop_duplicates(subset='title', keep='first')
len(data_cleaned)
def fun(data):
    data['title'] = data['add_to_cart'].apply(lambda x: x.get('title') if isinstance(x, dict) else None)
data = data.apply(fun)




import pandas as pd

# Load JSON data
data = pd.read_json('../udemy/courses/courses_scrape/courses_scrape/spiders/output.json')

# Function to extract title and print if list is empty
def title(dataa):
    if isinstance(dataa.get('add_to_cart'), list):
        if not dataa['add_to_cart']:  # Check if the list is empty
            print("Empty list found!")
            return None
        if dataa['add_to_cart'][0].get('title'):
            return dataa['add_to_cart'][0].get('title')
        else:
            print ("non value")
    return None  # Return None if 'add_to_cart' is missing or not a list

# Apply function to extract titles
karar = pd.DataFrame({'title': data.apply(title, axis=1)})
data.iloc[2]['curriculum_content'].keys()
data.iloc[2]['curriculum_content']['url']
# Display result
print(karar.head())

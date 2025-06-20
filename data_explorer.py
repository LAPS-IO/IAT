import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go
import plotly.express as px

import webbrowser
import pandas as pd
from os import listdir, getcwd, mkdir
from os.path import join, isdir, exists
from shutil import copy2

def get_projects_list():
    global projects_list

    projects_folder = join(getcwd(), 'main', 'assets')
    projects_list = [f for f in listdir(projects_folder) if isdir(join(projects_folder, f))]
    projects_list.sort()

def update_classes_project(batches_list, project_name):
    global classes_per_batch

    temp_classes_project = {}
    temp_classes_project_total = {}  # Store total counts (unfiltered)
    
    print("Update classes project:")
    print(f"Batches: {batches_list}")
    print(f"Classes per batch keys: {list(classes_per_batch.keys())}")

    for file in batches_list:
        if file in classes_per_batch:
            dict_temp = classes_per_batch[file]
            print(f"File {file}: {dict_temp}")
            for keys in dict_temp.keys():
                if keys not in temp_classes_project.keys():
                    temp_classes_project[keys] = dict_temp[keys]
                else:
                    temp_classes_project[keys] += dict_temp[keys]
        else:
            print(f"Warning: {file} not found in classes_per_batch")

    # Calculate total counts (unfiltered) for each class
    batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        dict_temp = dict(df_temp['manual_label'].value_counts())
        
        for keys in dict_temp.keys():
            if keys not in temp_classes_project_total.keys():
                temp_classes_project_total[keys] = dict_temp[keys]
            else:
                temp_classes_project_total[keys] += dict_temp[keys]

    temp_keys = list(temp_classes_project.keys())
    temp_keys.sort()
    counts = [temp_classes_project[k] for k in temp_keys]
    total_counts = [temp_classes_project_total[k] for k in temp_keys]
    classes_text = [{'label': f' {k} ({c}/{t})', 'value': 'enable'} for k, c, t in zip(temp_keys, counts, total_counts)]
    classes_list = [k for k in temp_keys]
    
    print(f"Final counts: {dict(zip(temp_keys, counts))}")
    print(f"Total counts: {dict(zip(temp_keys, total_counts))}")
    print()
    
    return classes_text, classes_list

def get_classes(batches_folder, batches_list):
    classes_list = []
    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        classes_list.extend(df_temp['manual_label'].unique())
    classes_list = list(set(classes_list))
    classes_list.sort()
    return classes_list

def get_histogram_data(batches_folder, batches_list, classes_list):
    """Generate histogram data for each class based on confidence distributions"""
    hist_data = {}
    
    for label in classes_list:
        all_confs = []
        
        for file in batches_list:
            file_path = join(batches_folder, file)
            df_temp = pd.read_csv(file_path)
            
            # Get confidence values for this class
            class_confs = df_temp[df_temp['manual_label'] == label]['manual_conf'].tolist()
            all_confs.extend(class_confs)
        
        if all_confs:
            hist_data[label] = all_confs
    
    return hist_data

def get_classes_per_batch(batches_folder, batches_list, classes_list, slider_values=None):
    global classes_per_batch, loadbar_classes

    print()
    print("Get classes per batch")
    print()
    
    if slider_values is not None:
        print(f"Slider values received in get_classes_per_batch: {slider_values}")
    else:
        print("No slider values provided, using default range [0.85, 1]")
    
    # Clear the classes_per_batch to ensure fresh calculation
    classes_per_batch = {}
    
    count = 0

    for file in batches_list:
        file_path = join(batches_folder, file)
        df_temp = pd.read_csv(file_path)
        
        # Initialize the file entry in classes_per_batch
        classes_per_batch[file] = {}

        print(len(slider_values) if slider_values else 0, len(classes_list))
        if slider_values is not None and len(slider_values) == len(classes_list):
            for i, label in enumerate(classes_list):
                filtered = df_temp[(df_temp['manual_label'] == label) & (df_temp['manual_conf'] >= slider_values[i][0]) & (df_temp['manual_conf'] < slider_values[i][1])]
                classes_per_batch[file][label] = len(filtered)
                print(f"Class {label}: {len(filtered)} rows (conf range: {slider_values[i][0]}-{slider_values[i][1]})")
        else:
            # Use default range [0.85, 1] when no slider values are provided
            for i, label in enumerate(classes_list):
                filtered = df_temp[(df_temp['manual_label'] == label) & (df_temp['manual_conf'] >= 0.85) & (df_temp['manual_conf'] < 1.0)]
                classes_per_batch[file][label] = len(filtered)
                print(f"Class {label}: {len(filtered)} rows (default conf range: 0.85-1.0)")

        count += 1
        loadbar_classes = 100*count/len(batches_list)

def save_dataset(output_folder, project_name, selected_classes, selected_batches, slider_values=None):
    global loadbar_save_dataset

    print("Save dataset")
    print(f"Selected classes: {selected_classes}")
    print(f"Selected batches: {selected_batches}")
    print(f"Slider values: {slider_values}")
    print()

    count = 0
    project_folder = join(getcwd(), 'main', 'assets', project_name)
    batches_folder = join(project_folder, 'dataframes')
    images_folder = join(project_folder, 'images')
    
    if not exists(output_folder):
        mkdir(output_folder)
    
    # Create folders for each class (both filtered and remaining)
    for label in selected_classes:
        if slider_values is not None and len(slider_values) == len(selected_classes):
            # Find the slider values for this class
            class_index = selected_classes.index(label)
            if class_index < len(slider_values) and slider_values[class_index] is not None:
                min_conf, max_conf = slider_values[class_index]
                # Create filtered folder
                if max_conf == 1.0:
                    filtered_folder_name = f"{label}>={int(min_conf*100)}"
                else:
                    filtered_folder_name = f"{label}>={int(min_conf*100)}and<{int(max_conf*100)}"
                filtered_label_path = join(output_folder, filtered_folder_name)
                if not exists(filtered_label_path):
                    mkdir(filtered_label_path)
                
                # Create remaining folder
                if max_conf == 1.0:
                    remaining_folder_name = f"{label}<{int(min_conf*100)}"
                else:
                    remaining_folder_name = f"{label}<{int(min_conf*100)}or>={int(max_conf*100)}"
                remaining_label_path = join(output_folder, remaining_folder_name)
                if not exists(remaining_label_path):
                    mkdir(remaining_label_path)
            else:
                # Fallback if no slider values
                label_path = join(output_folder, label)
                if not exists(label_path):
                    mkdir(label_path)
        else:
            # Fallback if no slider values
            label_path = join(output_folder, label)
            if not exists(label_path):
                mkdir(label_path)

    for batch in selected_batches:
        count += 1
        loadbar_save_dataset = 100*count/len(selected_batches) 

        batch_folder = batch[:-4]
        match = project_name + '.csv'
        if match in batch: #if project name is included in the csv name
            match_size = len(match) + 1
            batch_folder = batch[:-match_size]

        temp_images_folder = join(images_folder, batch_folder)
        batch_images_folder = join(temp_images_folder, listdir(temp_images_folder)[0])

        csv_path = join(batches_folder, batch)
        df_temp = pd.read_csv(csv_path)
        print(f"Batch {batch}: {len(df_temp)} total rows")
        
        # First filter by selected classes
        df_filtered = df_temp.loc[df_temp['manual_label'].isin(selected_classes)]
        print(f"After class filtering: {len(df_filtered)} rows")

        # Then apply slider filtering if available
        if slider_values is not None and len(slider_values) == len(selected_classes):
            print("Applying slider filtering...")
            filtered_rows = []
            remaining_rows = []
            
            for i, label in enumerate(selected_classes):
                if i < len(slider_values) and slider_values[i] is not None:
                    min_conf, max_conf = slider_values[i]
                    
                    # Get filtered images (within range)
                    class_filtered = df_filtered[
                        (df_filtered['manual_label'] == label) & 
                        (df_filtered['manual_conf'] >= min_conf) & 
                        (df_filtered['manual_conf'] < max_conf)
                    ]
                    print(f"Class {label}: {len(class_filtered)} rows (conf range: {min_conf}-{max_conf})")
                    filtered_rows.append(class_filtered)
                    
                    # Get remaining images (outside range)
                    class_remaining = df_filtered[
                        (df_filtered['manual_label'] == label) & 
                        ((df_filtered['manual_conf'] < min_conf) | (df_filtered['manual_conf'] > max_conf))
                    ]
                    print(f"Class {label}: {len(class_remaining)} rows (outside conf range: {min_conf}-{max_conf})")
                    remaining_rows.append(class_remaining)
            
            # Save filtered images
            if filtered_rows:
                df_filtered_combined = pd.concat(filtered_rows, ignore_index=True)
                print(f"Copying {len(df_filtered_combined)} filtered images from batch {batch}")
                for _, row in df_filtered_combined.iterrows():
                    name = row['names']
                    label = row['manual_label']
                    class_index = selected_classes.index(label)
                    min_conf, max_conf = slider_values[class_index]
                    if max_conf == 1.0:
                        filtered_folder_name = f"{label}>={int(min_conf*100)}"
                    else:
                        filtered_folder_name = f"{label}>={int(min_conf*100)}and<{int(max_conf*100)}"
                    filtered_label_path = join(output_folder, filtered_folder_name)
                    copy2(join(batch_images_folder, name), join(filtered_label_path, name))
            
            # Save remaining images
            if remaining_rows:
                df_remaining_combined = pd.concat(remaining_rows, ignore_index=True)
                print(f"Copying {len(df_remaining_combined)} remaining images from batch {batch}")
                for _, row in df_remaining_combined.iterrows():
                    name = row['names']
                    label = row['manual_label']
                    class_index = selected_classes.index(label)
                    min_conf, max_conf = slider_values[class_index]
                    if max_conf == 1.0:
                        remaining_folder_name = f"{label}<{int(min_conf*100)}"
                    else:
                        remaining_folder_name = f"{label}<{int(min_conf*100)}or>={int(max_conf*100)}"
                    remaining_label_path = join(output_folder, remaining_folder_name)
                    copy2(join(batch_images_folder, name), join(remaining_label_path, name))
        else:
            print("No slider filtering applied")
            print(f"Copying {len(df_filtered)} images from batch {batch}")
            for _, row in df_filtered.iterrows():
                name = row['names']
                label = row['manual_label']
                label_path = join(output_folder, label)
                copy2(join(batch_images_folder, name), join(label_path, name))
    
    loadbar_save_dataset = 0  
    return 0

def get_datasets_list():
    default_output_folder = join(getcwd(), 'output')
    datasets_list = [f for f in listdir(default_output_folder) if isdir(join(default_output_folder, f))]
    return datasets_list

projects_list = []
classes_per_batch = {}
loadbar_classes = 0
loadbar_save_dataset = 0
current_classes_list = []  # Store the current class names
get_projects_list()
datasets_list = get_datasets_list()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        }
    ]
)

app.layout = html.Div([
    ## header
    html.H1('ILT Data Explorer'),

    dcc.ConfirmDialog(
        id='confirm_save_dataset'
    ),

    html.Datalist(
        id='list_suggested_datasets', 
        children=[html.Option(value=name) for name in datasets_list]
    ),

    ## project selection
    dbc.Container(
        dbc.Row([
            dbc.Col(dcc.Dropdown(projects_list, '', id='dropdown_project', clearable=False), width={"size": 12}),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                [
                    dcc.Checklist(
                        options = [],
                        value = [],
                        id = 'checklist_batches',
                        labelStyle={'display': 'block'},
                        style={"height": 600, "overflow":"auto"}
                    )
                ], width = 6
            ),
            dbc.Col([
                dbc.Progress(value=0, id='progress_classes'),
                html.Div(id='slider-container')

                ], width = 6
            ),
        ]),
    ),

    dbc.Row([dbc.Col(html.Hr()),],),

    dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    value='dataset',
                    id='input_save_dataset',
                    type="text",
                    style={'width': '100%', 'background':'Floralwhite'},
                    list = 'list_suggested_datasets'
                ),
            ),
            dbc.Col([
                dbc.Button('Save Dataset', n_clicks=0, id='button_save_dataset', style={'background':'chocolate', 'width':'100%'}),
                dbc.Progress(value=0, id="progress_save_dataset"),
                dbc.Alert(children = '' , id="alert_save_dataset_problem", dismissable=True, is_open=False, color='warning', duration=5000),
                dbc.Alert(children = '' , id="alert_save_dataset_success", dismissable=True, is_open=False),
            ])
        ])
    ),


    dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
    dcc.Store(id='button_display_classes_nclicks', data=0),
    dcc.Store(id='dataset_name_ok', data=False),
    dcc.Store(id='save_dataset_result', data=0),
])

"""
    Updates the loadbar when the button_display_classes is pressed
"""
@app.callback(
    Output("progress_classes", "value"),
    Output("progress_save_dataset", "value"),
    Input("clock", "n_intervals"))
def progress_classes_update(n):
    global loadbar_classes, loadbar_save_dataset
    return (loadbar_classes, ), (loadbar_save_dataset, )

"""

"""
@app.callback(
    Output('slider-container', 'children'),
    Output('button_display_classes_nclicks', 'data'),
    Input('checklist_batches', 'value'),
    Input('slider-container', 'children'),
    Input({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),
    State('dropdown_project', 'value'),
    State('button_display_classes_nclicks', 'data')
)
def update_classes_list(checklist_value, slide_container, slider_values, project_name, prev_nclicks):
    global current_classes_list
    components = []
    
    print("=== update_classes_list called ===")
    print(f"checklist_value: {checklist_value}")
    print(f"slider_values: {slider_values}")
    print(f"project_name: {project_name}")
    
    # Check if triggered by batch selection or slider change
    batch_triggered = len(checklist_value) > 0
    
    # Check if triggered by slider change
    slider_triggered = any(value is not None for value in slider_values) if slider_values else False
    
    print(f"batch_triggered: {batch_triggered}")
    print(f"slider_triggered: {slider_triggered}")
    
    if batch_triggered or slider_triggered:
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
        classes_list = get_classes(batches_folder, checklist_value)
        get_classes_per_batch(batches_folder, checklist_value, classes_list, slider_values)
        classes_text, classes_list = update_classes_project(checklist_value, project_name)
        
        # Generate histogram data
        hist_data = get_histogram_data(batches_folder, checklist_value, classes_list)
        
        # Store the current class names globally
        current_classes_list = classes_list

        for i in range(len(classes_text)):
            # Use existing slider values or defaults
            slider_value = slider_values[i] if slider_values and i < len(slider_values) and slider_values[i] is not None else [0.85, 1]
            print(f"Using slider values: {slider_value}")
            
            # Create histogram for this class
            if classes_list[i] in hist_data:
                hist_fig = go.Figure()
                
                # Create exactly 20 bins with steps of 0.05
                bin_edges = [i * 0.05 for i in range(21)]  # 0, 0.05, 0.1, ..., 1.0
                
                # Debug: print the data for this class
                print(f"Histogram data for {classes_list[i]}: {hist_data[classes_list[i]]}")
                
                hist_fig.add_trace(go.Histogram(
                    x=hist_data[classes_list[i]],
                    xbins=dict(
                        start=0,
                        end=1.001,  # Slightly extend to ensure value 1 is included
                        size=0.05
                    ),
                    name=classes_list[i],
                    opacity=0.7,
                    marker_color='lightblue',
                    nbinsx=20  # Ensure exactly 20 bins
                ))
                
                # Add vertical lines for slider range
                hist_fig.add_vline(x=slider_value[0], line_dash="dash", line_color="red")
                hist_fig.add_vline(x=slider_value[1], line_dash="dash", line_color="red")
                
                hist_fig.update_layout(
                    height=48,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(
                        showticklabels=False, 
                        range=[-0.07, 1.07],
                        showgrid=False
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False
                    )
                )
            else:
                hist_fig = go.Figure()
                hist_fig.update_layout(
                    height=48,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=False, range=[-0.07, 1.07]),
                    yaxis=dict(showticklabels=False)
                )
            
            histogram = dcc.Graph(
                id={'type': 'dynamic-histogram', 'index': i},
                figure=hist_fig,
                config={'displayModeBar': False}
            )
            print(classes_text[i])

            checklist = dcc.Checklist(
                id={'type': 'dynamic-checklist', 'index': i},
                options=[classes_text[i]],
                value=['enable'],  # Default checked
                style={'marginLeft': '0px', 'width' : '45%'}
            )

            slider = dcc.RangeSlider(
                id={'type': 'dynamic-slider', 'index': i},
                min=0,
                max=1.0,
                step=0.05,
                value=slider_value,
                marks={0: '0', 1: '1'},
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                    "style": {"color": "DarkBlue", "fontSize": "20px"},
                }
            )

            row = html.Div([
                checklist,
                html.Div([
                    histogram,
                    html.Div(slider, style={'marginTop': '-16px'})
                ], style={'width': '45%'}),

            ], style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'gap': '20px',
                'marginBottom': '15px'
            })

            components.append(row)

        return components, prev_nclicks + 1 if batch_triggered else prev_nclicks
    
    return components, prev_nclicks

@app.callback(
    Output('checklist_batches', 'options'),
    Output('checklist_batches', 'value'),
    Output('input_save_dataset', 'value'),
    Input('dropdown_project', 'value')
)
def update_batches_list(project_name):
    global classes_per_batch, loadbar_classes, loadbar_save_dataset

    if project_name != '':
        batches_folder = join(getcwd(), 'main', 'assets', project_name, 'dataframes')
        batches_list = [f for f in listdir(batches_folder) if f[-4:] == '.csv']
        batches_list.sort()

        batches_text = []
        for i in range(len(batches_list)):
            batches_text.append({
                'label': ' ' + str(i+1) + ': ' + batches_list[i],
                'value': batches_list[i]
            })

        classes_per_batch = {}
        loadbar_classes = 0
        loadbar_save_dataset = 0

        return batches_text, batches_list, project_name + '_dataset'
    return [], [], 'dataset'


@app.callback(
    Output('alert_save_dataset_success', 'children'),
    Output('alert_save_dataset_success', 'is_open'),
    Input('confirm_save_dataset', 'submit_n_clicks'),
    State('dropdown_project', 'value'),
    State('input_save_dataset', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
    State({'type': 'dynamic-slider', 'index': dash.ALL}, 'value'),
)
def save_dataset_confirmed(nclicks, project_name, dataset_name, selected_batches, selected_classes, slider_values):
    global current_classes_list
    print("Save dataset confirmed")
    print()
    if nclicks:
        # Map checklist selections to actual class names
        actual_selected_classes = []
        if selected_classes and current_classes_list:
            for i, class_list in enumerate(selected_classes):
                if class_list and i < len(current_classes_list):  # Check if the inner list is not empty
                    actual_selected_classes.append(current_classes_list[i])
        
        default_output_folder = join(getcwd(), 'output')
        output_folder = join(default_output_folder, dataset_name)

        save_dataset(output_folder, project_name, actual_selected_classes, selected_batches, slider_values)
        return dataset_name + ' saved successfully', True
    return '', False

@app.callback(
    Output('confirm_save_dataset', 'message'),
    Output('confirm_save_dataset', 'displayed'),
    Input('dataset_name_ok', 'data'),
    State('input_save_dataset', 'value'),
)
def save_dataset_confirmation(name_ok, dataset_name):
    default_output_folder = join(getcwd(), 'output')
    output_folder = join(default_output_folder, dataset_name)

    if name_ok == True:
        if exists(output_folder):
            return dataset_name + ' already exists. Do you really want to merge these images into the existing dataset?', True
        else:
            return dataset_name + ' will be created. Proceed?', True
    return '', False


@app.callback(
    Output('alert_save_dataset_problem', 'children'),
    Output('alert_save_dataset_problem', 'is_open'),
    Output('dataset_name_ok', 'data'),
    Input('button_save_dataset', 'n_clicks'),
    State('dropdown_project', 'value'),
    State('checklist_batches', 'value'),
    State({'type': 'dynamic-checklist', 'index': dash.ALL}, 'value'),
)
def click_save_dataset(nclicks, project_name, selected_batches, selected_classes):
    global current_classes_list
    if nclicks > 0:
        default_output_folder = join(getcwd(), 'output')
        if not exists(default_output_folder):
            mkdir(default_output_folder)

        # Map checklist selections to actual class names for validation
        actual_selected_classes = []
        if selected_classes and current_classes_list:
            for i, class_list in enumerate(selected_classes):
                if class_list and i < len(current_classes_list):  # Check if the inner list is not empty
                    actual_selected_classes.append(current_classes_list[i])

        if project_name == '':
            return 'No project was selected', True, False
        elif len(selected_batches) == 0:
            return 'No batches were selected', True, False
        elif len(actual_selected_classes) == 0:
            return 'No classes were selected', True, False
        else:
            return '', False, True
    return '', False, False

port = 8030
opened = False

if not opened:
    webbrowser.open('http://127.0.0.1:' + str(port) + '/', new=2, autoraise=True)
    opened = True

if __name__ == '__main__':
    app.title = 'ILT Data Explorer'
    app.run_server(debug=False, port=port)

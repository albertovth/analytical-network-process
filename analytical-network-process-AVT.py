#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:05:57 2024

@author: albertovth
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def create_supermatrix(node_list, relationships, weights):
    n = len(node_list)
    supermatrix = np.zeros((n, n))
    node_index = {node: index for index, node in enumerate(node_list)}
    
    for relationship, weight in weights.items():
        source, target = relationship
        i, j = node_index[source], node_index[target]
        supermatrix[i, j] = weight
    
    return supermatrix

def normalize_supermatrix(supermatrix):
    normalized_matrix = np.zeros_like(supermatrix)
    column_sums = np.sum(supermatrix, axis=0)
    
    for col_index, sum in enumerate(column_sums):
        if sum != 0:
            normalized_matrix[:, col_index] = supermatrix[:, col_index] / sum
    return normalized_matrix

def calculate_priority_vector(supermatrix):
    eigenvalues, eigenvectors = np.linalg.eig(supermatrix)
    largest_index = np.argmax(eigenvalues)
    priority_vector = eigenvectors[:, largest_index].real
    priority_vector /= np.sum(priority_vector)
    return priority_vector

def display_consolidated_weights(priority_vector, node_list):
    sorted_indices = np.argsort(-priority_vector)
    consolidated_weights = pd.DataFrame({
        'Node': [node_list[i] for i in sorted_indices],
        'Weight': priority_vector[sorted_indices]
    })
    
    consolidated_weights.set_index('Node', inplace=True)

    formatted_df = consolidated_weights.style.format({'Weight': '{:.2f}'})
   
    st.write("Network Synthetic Weights")
    st.dataframe(formatted_df)


def input_network_relationships(node_list):
    relationships = []
    weights = {}
    num_relationships = st.number_input('How many relationships do you want to define?', min_value=0, value=0, step=1)
    
    for i in range(num_relationships):
        source = st.selectbox(f'Select the source node for relationship {i+1}', options=node_list, index=0, key=f'source_{i}')
        target = st.selectbox(f'Select the target node for relationship {i+1}', options=node_list, index=0, key=f'target_{i}')
        weight = st.number_input(f'Weight for relationship {i+1}', value=1.0, min_value=0.0, max_value=None, key=f'weight_{i}')
        
        relationships.append((source, target))
        weights[(source, target)] = weight
    
    return relationships, weights

def draw_network_diagram(diagram_title, main_criteria_node_list, criteria_node_list, alternative_node_list, relationships, weights):
    G = nx.DiGraph()

    all_nodes = main_criteria_node_list + criteria_node_list + alternative_node_list
    for node in all_nodes:
        G.add_node(node)
    for (source, target), weight in weights.items():
        G.add_edge(source, target, weight=weight)

    pos = {}
    
    criteria_spacing = 1

    total_width = (len(criteria_node_list) - 1) * criteria_spacing
   
    main_criteria_spacing = 1  
    main_criteria_width = (len(main_criteria_node_list) - 1) * main_criteria_spacing
   
    main_criteria_x_start = (total_width - main_criteria_width) / 2
   
    for index, node in enumerate(main_criteria_node_list):
       pos[node] = np.array([main_criteria_x_start + index * main_criteria_spacing, 2])  

    criteria_heights = [1.5, 1.2, 1.5]  
    criteria_spacing = 1*(max(len(main_criteria_node_list), len(alternative_node_list)) / max(1, len(criteria_node_list)))
    for index, node in enumerate(criteria_node_list):
        pos[node] = np.array([index * criteria_spacing, criteria_heights[index % len(criteria_heights)]])

    alternative_heights = [0.5, 0.2, 0.5]  
    alternative_spacing = 1*(max(len(main_criteria_node_list), len(criteria_node_list)) / max(1, len(alternative_node_list)))
    for index, node in enumerate(alternative_node_list):
        pos[node] = np.array([index * alternative_spacing, alternative_heights[index % len(alternative_heights)]])

    fig, ax = plt.subplots(figsize=(12,8))
    
    node_size_main = 2000
    node_size_network = 1800
   
    nx.draw_networkx_nodes(G, pos, nodelist=main_criteria_node_list, node_color='red', node_size=node_size_main, ax=ax,alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=criteria_node_list, node_color='skyblue', node_size=node_size_network, ax=ax,alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=alternative_node_list, node_color='lightgreen', node_size=node_size_network, ax=ax,alpha=0.5)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
    
    label_pos = {node: (position[0], position[1] - 0.1) for node, position in pos.items()} 
    
    nx.draw_networkx_labels(G, label_pos, font_weight='bold', ax=ax)

    edge_labels = {}
    for (source, target), weight in weights.items():
        if (target, source) in weights and (source, target) not in edge_labels:
            edge_labels[(source, target)] = f"{weight} / {weights[(target, source)]}"
        elif (source, target) not in edge_labels:
            edge_labels[(source, target)] = weight
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    plt.title(diagram_title)
    plt.axis('off')
    st.pyplot(fig)

st.set_page_config(layout="wide")

def app():
    st.title('Analytical Network Process (ANP) Application')

    col1, col2=st.columns(2)

    with col1:
        diagram_title = st.text_input('Diagram Title', 'Network Process Diagram')
        main_criterion_input = st.text_input('Enter the main criterion for the network analysis', 'Main Criterion')
        criteria_input = st.text_input('Enter criteria separated by comma', 'Criterion 1, Criterion 2')
        alternatives_input = st.text_input('Enter alternatives separated by comma', 'Alternative 1, Alternative 2')
    
        main_criteria = [a.strip() for a in main_criterion_input.split(',')]  # Selv om vi forventer kun én verdi her
        criteria = [c.strip() for c in criteria_input.split(',')]
        alternatives = [a.strip() for a in alternatives_input.split(',')]
        main_criteria_node_list= main_criteria
        criteria_node_list = criteria
        alternative_node_list = alternatives
    
        node_list=main_criteria_node_list+criteria_node_list+alternative_node_list
    
        relationships, weights = input_network_relationships(node_list)            
        
    with col2:        
        draw_network_diagram(diagram_title, main_criteria_node_list, criteria_node_list, alternative_node_list, relationships, weights)
        
        if st.button('Create Supermatrix and Analyze'):
            supermatrix = create_supermatrix(node_list, relationships, weights)
            normalized_matrix = normalize_supermatrix(supermatrix)
            priority_vector = calculate_priority_vector(normalized_matrix)
            st.markdown('''
            ### Network Consolidated Weights
            ''')
            display_consolidated_weights(priority_vector, node_list)
            st.markdown('''
            ### Normalized Supermatrix
            ''')
            st.write(normalized_matrix)
        
    
if __name__ == '__main__':
    app()

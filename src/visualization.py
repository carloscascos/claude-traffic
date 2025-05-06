"""
Módulo para la visualización de datos de tráfico marítimo.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import numpy as np
from datetime import datetime, timedelta
import os

# Configuración de estilo para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def plot_vessel_routes(routes_df, map_center=None):
    """
    Crea un mapa interactivo con las rutas de los buques.
    
    Args:
        routes_df (pandas.DataFrame): DataFrame con las coordenadas de las rutas
                                     (debe contener lat, lng, imo, vessel_name)
        map_center (tuple, opcional): Coordenadas del centro del mapa (lat, lng). 
                                     Defaults to None.
    
    Returns:
        folium.Map: Mapa interactivo con las rutas
    """
    if map_center is None:
        # Si no se proporciona un centro, usar el promedio de las coordenadas
        avg_lat = routes_df['lat'].mean()
        avg_lng = routes_df['lng'].mean()
        map_center = (avg_lat, avg_lng)
    
    # Crear mapa
    vessel_map = folium.Map(location=map_center, zoom_start=8, 
                            tiles='cartodbpositron')
    
    # Agrupar por IMO para obtener las rutas individuales de cada buque
    for imo, group in routes_df.groupby('imo'):
        # Obtener el nombre del buque
        vessel_name = group['vessel_name'].iloc[0] if 'vessel_name' in group.columns else f"IMO: {imo}"
        
        # Obtener las coordenadas de la ruta
        route_points = list(zip(group['lat'], group['lng']))
        
        # Añadir la ruta al mapa
        folium.PolyLine(
            route_points,
            color=get_random_color(),
            weight=3,
            opacity=0.8,
            popup=vessel_name
        ).add_to(vessel_map)
        
        # Añadir marcadores para el inicio y fin de la ruta
        folium.Marker(
            route_points[0],
            popup=f"Inicio: {vessel_name}",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(vessel_map)
        
        folium.Marker(
            route_points[-1],
            popup=f"Fin: {vessel_name}",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(vessel_map)
    
    return vessel_map


def plot_port_traffic_heatmap(port_calls_df, map_center=None, radius=15):
    """
    Crea un mapa de calor del tráfico portuario.
    
    Args:
        port_calls_df (pandas.DataFrame): DataFrame con las llamadas a puerto
                                        (debe contener lat, lng, portname)
        map_center (tuple, opcional): Coordenadas del centro del mapa (lat, lng).
                                    Defaults to None.
        radius (int, opcional): Radio de los puntos del mapa de calor. Defaults to 15.
    
    Returns:
        folium.Map: Mapa interactivo con el heatmap
    """
    if map_center is None:
        # Si no se proporciona un centro, usar el promedio de las coordenadas
        avg_lat = port_calls_df['lat'].mean()
        avg_lng = port_calls_df['lng'].mean()
        map_center = (avg_lat, avg_lng)
    
    # Crear mapa
    heat_map = folium.Map(location=map_center, zoom_start=4, 
                          tiles='cartodbpositron')
    
    # Contar el número de visitas por puerto
    port_counts = port_calls_df.groupby(['portname', 'lat', 'lng']).size().reset_index(name='count')
    
    # Preparar datos para el heatmap - incluir peso basado en el conteo
    heat_data = [[row['lat'], row['lng'], row['count']] for _, row in port_counts.iterrows()]
    
    # Añadir capa de calor
    HeatMap(heat_data, radius=radius).add_to(heat_map)
    
    # Añadir marcadores de puertos
    marker_cluster = MarkerCluster().add_to(heat_map)
    
    for _, row in port_counts.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"{row['portname']}: {row['count']} visitas",
            icon=folium.Icon(color='blue', icon='ship', prefix='fa')
        ).add_to(marker_cluster)
    
    return heat_map


def plot_vessel_type_distribution(df, title="Distribución de Tipos de Buques"):
    """
    Crea un gráfico de pastel con la distribución de tipos de buques.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos (debe contener columna 'Type' o 'vessel_type')
        title (str, opcional): Título del gráfico. Defaults to "Distribución de Tipos de Buques".
    
    Returns:
        matplotlib.figure.Figure: Figura con el gráfico
    """
    type_column = 'vessel_type' if 'vessel_type' in df.columns else 'Type'
    
    if type_column not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{type_column}'")
    
    # Contar tipos de buques
    type_counts = df[type_column].value_counts()
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        type_counts, 
        autopct='%1.1f%%',
        textprops={'color': "w", 'fontsize': 12},
        shadow=True,
        startangle=90
    )
    
    # Añadir leyenda
    ax.legend(
        wedges, 
        type_counts.index,
        title="Tipos de Buques",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(title)
    
    return fig


def plot_monthly_port_calls(df, port_name=None, date_column='start'):
    """
    Crea un gráfico de líneas con el número de llamadas a puerto por mes.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos (debe contener columna de fecha)
        port_name (str, opcional): Nombre del puerto para filtrar. Si es None, se consideran todos.
        date_column (str, opcional): Nombre de la columna de fecha. Defaults to 'start'.
    
    Returns:
        matplotlib.figure.Figure: Figura con el gráfico
    """
    if date_column not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{date_column}'")
    
    # Convertir a datetime si es necesario
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Filtrar por puerto si se especifica
    if port_name:
        if 'portname' not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'portname'")
        data = df[df['portname'] == port_name].copy()
    else:
        data = df.copy()
    
    # Crear columna de mes
    data['month'] = data[date_column].dt.to_period('M')
    
    # Contar llamadas por mes
    monthly_counts = data.groupby('month').size().reset_index(name='count')
    
    # Convertir Period a datetime para el gráfico
    monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='month', y='count', data=monthly_counts, marker='o', linewidth=2, ax=ax)
    
    # Formatear eje x
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Añadir títulos y etiquetas
    port_title = f" en {port_name}" if port_name else ""
    ax.set_title(f"Llamadas a Puerto{port_title} por Mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Número de Llamadas")
    
    plt.tight_layout()
    return fig


def get_random_color():
    """
    Genera un color aleatorio en formato hexadecimal.
    
    Returns:
        str: Color en formato hexadecimal
    """
    r = int(np.random.random() * 255)
    g = int(np.random.random() * 255)
    b = int(np.random.random() * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


def save_map(map_object, filename):
    """
    Guarda un mapa de folium como archivo HTML.
    
    Args:
        map_object (folium.Map): Mapa a guardar
        filename (str): Nombre del archivo (sin extensión)
    """
    wd = os.getcwd()
    path = f"{wd}{filename}"

    map_object.save(f"{path}.html")


def save_figure(fig, filename, dpi=300):

    import os
    """
    Guarda una figura de matplotlib como archivo PNG.
    
    Args:
        fig (matplotlib.figure.Figure): Figura a guardar
        filename (str): Nombre del archivo (sin extensión)
        dpi (int, opcional): Resolución de la imagen. Defaults to 300.
    """
    fig.savefig(f"{os.getcwd()}{filename}.png", dpi=dpi, bbox_inches='tight')

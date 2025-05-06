"""
Script para generar ejemplos de visualización con datos reales.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from database import query_to_dataframe
from visualization import (plot_vessel_type_distribution, plot_monthly_port_calls,
                          plot_port_traffic_heatmap, save_map, save_figure)

# Configuración para gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Crear directorio para ejemplos si no existe
os.makedirs('../docs/examples', exist_ok=True)


def generate_vessel_type_chart():
    """
    Genera un gráfico de distribución de tipos de buques.
    """
    print("Generando gráfico de distribución de tipos de buques...")
    
    # Consulta para obtener los tipos de buques
    query = """
    SELECT Type, COUNT(*) as count
    FROM imo
    WHERE Type IS NOT NULL
    GROUP BY Type
    ORDER BY count DESC
    LIMIT 15
    """
    
    df = query_to_dataframe(query)
    
    if df.empty:
        print("No se encontraron datos de tipos de buques.")
        return
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear gráfico de barras
    bars = sns.barplot(x='Type', y='count', data=df, ax=ax)
    
    # Ajustar etiquetas
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribución de Tipos de Buques", fontsize=16)
    plt.xlabel("Tipo de Buque", fontsize=14)
    plt.ylabel("Cantidad", fontsize=14)
    
    # Añadir valores en las barras
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'),
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')
    
    plt.tight_layout()
    
    # Guardar gráfico
    save_figure(fig, '/docs/examples/vessel_types_distribution')
    print("Gráfico guardado en: ../docs/examples/vessel_types_distribution.png")


def generate_port_activity_chart(port_name="BARCELONA"):
    """
    Genera un gráfico de actividad mensual en un puerto.
    
    Args:
        port_name (str, opcional): Nombre del puerto a analizar. Defaults to "BARCELONA".
    """
    print(f"Generando gráfico de actividad mensual en {port_name}...")
    
    # Calcular fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Último año
    
    # Consulta para obtener las escalas en el puerto
    query = """
    SELECT start, end, imo, portname
    FROM escalas
    WHERE portname = %s
    AND start BETWEEN %s AND %s
    """
    
    df = query_to_dataframe(query, (port_name, start_date.strftime('%Y-%m-%d'), 
                                    end_date.strftime('%Y-%m-%d')))
    
    if df.empty:
        print(f"No se encontraron datos para el puerto {port_name}.")
        return
    
    # Crear gráfico
    fig = plot_monthly_port_calls(df, port_name)
    
    # Guardar gráfico
    save_figure(fig, f'/docs/examples/port_activity_{port_name}')
    print(f"Gráfico guardado en: ../docs/examples/port_activity_{port_name}.png")


def generate_port_traffic_map(country="SPAIN", limit=500):
    """
    Genera un mapa de calor del tráfico portuario en un país.
    
    Args:
        country (str, opcional): Nombre del país a analizar. Defaults to "SPAIN".
        limit (int, opcional): Límite de registros a recuperar. Defaults to 500.
    """
    print(f"Generando mapa de tráfico portuario en {country}...")
    
    # Calcular fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Último año
    
    # Consulta para obtener las escalas en el país
    query = """
    SELECT start, end, imo, portname, lat, lng
    FROM escalas
    WHERE country = %s
    AND start BETWEEN %s AND %s
    AND lat IS NOT NULL AND lng IS NOT NULL
    LIMIT %s
    """
    
    df = query_to_dataframe(query, (country, start_date.strftime('%Y-%m-%d'), 
                                   end_date.strftime('%Y-%m-%d'), limit))
    
    if df.empty:
        print(f"No se encontraron datos para el país {country}.")
        return
    
    # Crear mapa
    port_map = plot_port_traffic_heatmap(df)
    
    # Guardar mapa
    save_map(port_map, f'/docs/examples/port_traffic_{country}')
    print(f"Mapa guardado en: /docs/examples/port_traffic_{country}.html")


def generate_vessel_flag_chart():
    """
    Genera un gráfico de distribución de banderas de buques.
    """
    print("Generando gráfico de distribución de banderas...")
    
    # Consulta para obtener las banderas
    query = """
    SELECT FLAG, COUNT(*) as count
    FROM imo
    WHERE FLAG IS NOT NULL
    GROUP BY FLAG
    ORDER BY count DESC
    LIMIT 15
    """
    
    df = query_to_dataframe(query)
    
    if df.empty:
        print("No se encontraron datos de banderas de buques.")
        return
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear gráfico de barras
    bars = sns.barplot(x='FLAG', y='count', data=df, palette='viridis', ax=ax)
    
    # Ajustar etiquetas
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribución de Banderas de Buques", fontsize=16)
    plt.xlabel("Bandera", fontsize=14)
    plt.ylabel("Cantidad", fontsize=14)
    
    # Añadir valores en las barras
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'),
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')
    
    plt.tight_layout()
    
    # Guardar gráfico
    save_figure(fig, '/docs/examples/vessel_flags_distribution')
    print("Gráfico guardado en: /docs/examples/vessel_flags_distribution.png")


def main():
    """
    Función principal para generar ejemplos.
    """
    print("Generando ejemplos de visualización...")
    
    try:
        # Generar ejemplos
        generate_vessel_type_chart()
        generate_port_activity_chart("BARCELONA")
        generate_port_traffic_map("SPAIN")
        generate_vessel_flag_chart()
        
        print("\nGeneración de ejemplos completada con éxito.")
    except Exception as e:
        print(f"Error al generar ejemplos: {e}")


if __name__ == "__main__":
    main()

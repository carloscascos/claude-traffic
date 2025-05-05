"""
Módulo para el análisis de datos de tráfico marítimo.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from database import query_to_dataframe

# Configuración para visualizaciones
sns.set(style="whitegrid")


def calculate_port_statistics(port_name, start_date=None, end_date=None):
    """
    Calcula estadísticas de un puerto específico.
    
    Args:
        port_name (str): Nombre del puerto
        start_date (str, opcional): Fecha de inicio (formato: 'YYYY-MM-DD'). 
                                   Defaults to None (último año).
        end_date (str, opcional): Fecha de fin (formato: 'YYYY-MM-DD'). 
                                 Defaults to None (fecha actual).
    
    Returns:
        dict: Diccionario con estadísticas del puerto
    """
    if not start_date:
        # Si no se proporciona fecha de inicio, usar el último año
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if not end_date:
        # Si no se proporciona fecha de fin, usar la fecha actual
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Consulta para obtener las escalas en el puerto
    query = """
    SELECT e.*, i.NAME, i.FLAG, i.Type, i.Category, i.DWT, i.GT
    FROM escalas e
    JOIN imo i ON e.imo = i.IMO
    WHERE e.portname = %s
    AND e.start BETWEEN %s AND %s
    """
    
    df = query_to_dataframe(query, (port_name, start_date, end_date))
    
    if df.empty:
        return {
            "port_name": port_name,
            "error": "No se encontraron datos para el puerto y fechas especificadas."
        }
    
    # Convertir columnas de fecha si no son datetime
    if df['start'].dtype != 'datetime64[ns]':
        df['start'] = pd.to_datetime(df['start'])
    if df['end'].dtype != 'datetime64[ns]':
        df['end'] = pd.to_datetime(df['end'])
    
    # Calcular estadísticas
    total_calls = len(df)
    unique_vessels = df['imo'].nunique()
    avg_stay_duration = df['duration'].mean()
    
    # Calcular estadísticas por tipo de buque
    vessel_type_counts = df['Type'].value_counts().to_dict()
    top_flags = df['FLAG'].value_counts().head(5).to_dict()
    
    # Calcular tendencia mensual
    df['month'] = df['start'].dt.to_period('M')
    monthly_calls = df.groupby('month').size()
    monthly_trend = monthly_calls.pct_change().mean() * 100  # Cambio porcentual promedio
    
    # Calcular estadísticas de tamaño (DWT, GT)
    avg_dwt = df['DWT'].mean()
    avg_gt = df['GT'].mean()
    
    # Origen y destino principales
    top_prev_ports = df['prev_port'].value_counts().head(5).to_dict()
    top_next_ports = df['next_port'].value_counts().head(5).to_dict()
    
    # Devolver resultados
    return {
        "port_name": port_name,
        "period": {
            "start_date": start_date,
            "end_date": end_date
        },
        "general_stats": {
            "total_calls": total_calls,
            "unique_vessels": unique_vessels,
            "avg_stay_duration": avg_stay_duration
        },
        "vessel_types": vessel_type_counts,
        "top_flags": top_flags,
        "monthly_trend": monthly_trend,
        "size_stats": {
            "avg_dwt": avg_dwt,
            "avg_gt": avg_gt
        },
        "connections": {
            "top_origins": top_prev_ports,
            "top_destinations": top_next_ports
        }
    }


def analyze_vessel_patterns(imo, period_days=365):
    """
    Analiza los patrones de movimiento de un buque específico.
    
    Args:
        imo (int): Número IMO del buque
        period_days (int, opcional): Número de días a analizar. Defaults to 365.
    
    Returns:
        dict: Diccionario con análisis de patrones del buque
    """
    # Calcular fecha de inicio
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Consulta para obtener las escalas del buque
    query = """
    SELECT e.*, i.NAME, i.FLAG, i.Type
    FROM escalas e
    JOIN imo i ON e.imo = i.IMO
    WHERE e.imo = %s
    AND e.start >= %s
    ORDER BY e.start ASC
    """
    
    df = query_to_dataframe(query, (imo, start_date.strftime('%Y-%m-%d')))
    
    if df.empty:
        return {
            "imo": imo,
            "error": "No se encontraron datos para el buque en el período especificado."
        }
    
    # Información básica del buque
    vessel_name = df['NAME'].iloc[0]
    vessel_type = df['Type'].iloc[0]
    vessel_flag = df['FLAG'].iloc[0]
    
    # Convertir columnas de fecha si no son datetime
    if df['start'].dtype != 'datetime64[ns]':
        df['start'] = pd.to_datetime(df['start'])
    if df['end'].dtype != 'datetime64[ns]':
        df['end'] = pd.to_datetime(df['end'])
    
    # Calcular tiempo en puerto vs navegación
    total_port_time = df['duration'].sum()
    
    # Calcular tiempo total entre la primera y última escala
    total_period = (df['start'].iloc[-1] - df['start'].iloc[0]).total_seconds() / 3600  # en horas
    
    # Tiempo en navegación (aproximado)
    # Convertir total_port_time a float para evitar error de tipos
    navigation_time = total_period - float(total_port_time)
    
    # Puertos más visitados
    top_ports = df['portname'].value_counts().head(5).to_dict()
    
    # Países más visitados
    top_countries = df['country'].value_counts().head(5).to_dict()
    
    # Calcular distancia total recorrida
    total_distance = df['prev_leg'].sum()
    total_distance = float(total_distance) if total_distance is not None else 0.0
    
    # Calcular velocidad promedio
    avg_speed = df['speed'].mean()
    avg_speed = float(avg_speed) if avg_speed is not None else 0.0
    
    # Detección de rutas frecuentes
    port_pairs = []
    for i in range(len(df) - 1):
        port_pairs.append((df['portname'].iloc[i], df['portname'].iloc[i+1]))
    
    route_counts = pd.Series(port_pairs).value_counts().head(3).to_dict()
    frequent_routes = [{"route": f"{origin} → {destination}", "count": count} 
                      for (origin, destination), count in route_counts.items()]
    
    # Devolver resultados
    return {
        "vessel_info": {
            "imo": imo,
            "name": vessel_name,
            "type": vessel_type,
            "flag": vessel_flag
        },
        "activity_summary": {
            "total_port_calls": len(df),
            "total_port_time_hours": float(total_port_time),
            "total_navigation_time_hours": float(navigation_time),
            "port_time_percentage": float(float(total_port_time) / total_period * 100) if total_period > 0 else 0,
            "navigation_time_percentage": float(navigation_time / total_period * 100) if total_period > 0 else 0
        },
        "top_ports": top_ports,
        "top_countries": top_countries,
        "navigation_stats": {
            "total_distance_nm": float(total_distance),
            "avg_speed_knots": float(avg_speed)
        },
        "frequent_routes": frequent_routes
    }


def compare_ports(port_names, period_days=180):
    """
    Compara estadísticas entre varios puertos.
    
    Args:
        port_names (list): Lista de nombres de puertos a comparar
        period_days (int, opcional): Número de días a considerar. Defaults to 180.
    
    Returns:
        pandas.DataFrame: DataFrame con la comparación de puertos
    """
    # Calcular fecha de inicio
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    
    # Inicializar DataFrame de resultados
    results = []
    
    for port in port_names:
        # Obtener estadísticas del puerto
        port_stats = calculate_port_statistics(
            port, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Si hay error, continuar con el siguiente puerto
        if 'error' in port_stats:
            continue
        
        # Extraer estadísticas clave
        result = {
            'port_name': port,
            'total_calls': port_stats['general_stats']['total_calls'],
            'unique_vessels': port_stats['general_stats']['unique_vessels'],
            'avg_stay_duration': port_stats['general_stats']['avg_stay_duration'],
            'avg_dwt': port_stats['size_stats']['avg_dwt'],
            'avg_gt': port_stats['size_stats']['avg_gt'],
            'monthly_trend': port_stats['monthly_trend']
        }
        
        results.append(result)
    
    # Convertir a DataFrame
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def analyze_regional_traffic(country, start_date=None, end_date=None, limit=1000):
    """
    Analiza el tráfico marítimo de una región o país.
    
    Args:
        country (str): Nombre del país o región
        start_date (str, opcional): Fecha de inicio (formato: 'YYYY-MM-DD'). 
                                   Defaults to None (último año).
        end_date (str, opcional): Fecha de fin (formato: 'YYYY-MM-DD'). 
                                 Defaults to None (fecha actual).
        limit (int, opcional): Límite de registros a recuperar. Defaults to 1000.
    
    Returns:
        dict: Diccionario con análisis del tráfico regional
    """
    if not start_date:
        # Si no se proporciona fecha de inicio, usar el último año
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if not end_date:
        # Si no se proporciona fecha de fin, usar la fecha actual
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Consulta para obtener las escalas en la región
    query = """
    SELECT e.*, i.NAME, i.FLAG, i.Type, i.Category, i.DWT, i.GT, p.zone
    FROM escalas e
    JOIN imo i ON e.imo = i.IMO
    LEFT JOIN ports p ON e.portname = p.portname
    WHERE e.country = %s
    AND e.start BETWEEN %s AND %s
    LIMIT %s
    """
    
    df = query_to_dataframe(query, (country, start_date, end_date, limit))
    
    if df.empty:
        return {
            "country": country,
            "error": "No se encontraron datos para el país y fechas especificadas."
        }
    
    # Convertir columnas de fecha si no son datetime
    if df['start'].dtype != 'datetime64[ns]':
        df['start'] = pd.to_datetime(df['start'])
    if df['end'].dtype != 'datetime64[ns]':
        df['end'] = pd.to_datetime(df['end'])
    
    # Principales puertos de la región
    top_ports = df['portname'].value_counts().head(10).to_dict()
    
    # Tipos de buques más comunes
    vessel_types = df['Type'].value_counts().head(10).to_dict()
    
    # Banderas más comunes
    flags = df['FLAG'].value_counts().head(10).to_dict()
    
    # Calcular tendencia mensual
    df['month'] = df['start'].dt.to_period('M')
    monthly_calls = df.groupby('month').size().reset_index(name='count')
    monthly_calls['month'] = monthly_calls['month'].dt.to_timestamp()
    
    # Calcular movimiento entre zonas
    zone_movement = {}
    if 'zone' in df.columns:
        # Agrupar por zona y contar escalas
        zone_counts = df['zone'].value_counts().to_dict()
        
        # Analizar movimientos entre zonas
        prev_zones = df.groupby('prev_port')['zone'].first().to_dict()
        df['prev_zone'] = df['prev_port'].map(prev_zones)
        
        # Contar movimientos entre zonas
        zone_pairs = df[['prev_zone', 'zone']].dropna().value_counts().head(10).to_dict()
        zone_movement = {f"{z1} → {z2}": count for (z1, z2), count in zone_pairs.items()}
    
    # Devolver resultados
    return {
        "country": country,
        "period": {
            "start_date": start_date,
            "end_date": end_date
        },
        "top_ports": top_ports,
        "vessel_types": vessel_types,
        "flags": flags,
        "monthly_trend": monthly_calls.to_dict(orient='records'),
        "zone_movement": zone_movement
    }


def get_vessel_types_distribution():
    """
    Obtiene la distribución global de tipos de buques en la base de datos.
    
    Returns:
        pandas.DataFrame: DataFrame con la distribución de tipos de buques
    """
    query = """
    SELECT Type, COUNT(*) as count
    FROM imo
    WHERE Type IS NOT NULL
    GROUP BY Type
    ORDER BY count DESC
    """
    
    return query_to_dataframe(query)

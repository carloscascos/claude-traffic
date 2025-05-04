"""
Módulo para la conexión y operaciones con la base de datos.
"""
import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Cargar variables de entorno
# Cargar variables de entorno
# Cargar variables de entorno
load_dotenv()

# ...existing code...
def get_db_connection():
    """
    Establece una conexión con la base de datos utilizando variables de entorno.
    
    Returns:
        mysql.connector.connection: Objeto de conexión a la base de datos
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "imo")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def execute_query(query, params=None, fetch=True):
    """
    Ejecuta una consulta SQL en la base de datos.
    
    Args:
        query (str): Consulta SQL a ejecutar
        params (tuple, opcional): Parámetros para la consulta. Defaults to None.
        fetch (bool, opcional): Indica si se debe recuperar los resultados. Defaults to True.
        
    Returns:
        list/bool: Resultados de la consulta o True/False en caso de éxito/error
    """
    connection = get_db_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            result = cursor.fetchall()
            return result
        else:
            connection.commit()
            return True
    except Error as e:
        print(f"Error al ejecutar la consulta: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def query_to_dataframe(query, params=None):
    """
    Ejecuta una consulta SQL y devuelve los resultados como un DataFrame de pandas.
    
    Args:
        query (str): Consulta SQL a ejecutar
        params (tuple, opcional): Parámetros para la consulta. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame con los resultados de la consulta
    """
    result = execute_query(query, params)
    if result:
        return pd.DataFrame(result)
    return pd.DataFrame()

def get_vessel_info(imo):
    """
    Obtiene información detallada de un buque por su número IMO.
    
    Args:
        imo (int): Número IMO del buque
        
    Returns:
        dict: Información del buque
    """
    query = "SELECT * FROM imo WHERE IMO = %s"
    result = execute_query(query, (imo,))
    return result[0] if result else None

def get_vessel_port_calls(imo, limit=10):
    """
    Obtiene las últimas escalas de un buque por su número IMO.
    
    Args:
        imo (int): Número IMO del buque
        limit (int, opcional): Número máximo de resultados. Defaults to 10.
        
    Returns:
        pandas.DataFrame: DataFrame con las escalas del buque
    """
    query = """
    SELECT start, end, duration, portname, country, 
           prev_port, next_port, prev_leg, next_leg
    FROM escalas 
    WHERE imo = %s 
    ORDER BY start DESC 
    LIMIT %s
    """
    return query_to_dataframe(query, (imo, limit))

def get_port_traffic(port_name, start_date=None, end_date=None, limit=50):
    """
    Obtiene el tráfico de buques en un puerto específico.
    
    Args:
        port_name (str): Nombre del puerto
        start_date (str, opcional): Fecha de inicio (formato: 'YYYY-MM-DD'). Defaults to None.
        end_date (str, opcional): Fecha de fin (formato: 'YYYY-MM-DD'). Defaults to None.
        limit (int, opcional): Número máximo de resultados. Defaults to 50.
        
    Returns:
        pandas.DataFrame: DataFrame con el tráfico del puerto
    """
    conditions = ["portname = %s"]
    params = [port_name]
    
    if start_date:
        conditions.append("start >= %s")
        params.append(start_date)
    
    if end_date:
        conditions.append("end <= %s")
        params.append(end_date)
    
    where_clause = " AND ".join(conditions)
    params.append(limit)
    
    query = f"""
    SELECT e.start, e.end, e.duration, e.imo, i.NAME as vessel_name, 
           i.FLAG as flag, i.Type as vessel_type, i.GT as gross_tonnage
    FROM escalas e
    JOIN imo i ON e.imo = i.IMO
    WHERE {where_clause}
    ORDER BY e.start DESC
    LIMIT %s
    """
    
    return query_to_dataframe(query, tuple(params))


if __name__ == "__main__":
    # Ejemplo de uso
    vessel_info = get_vessel_info(9769300)
    print(vessel_info)
    
    vessel_calls = get_vessel_port_calls(9769300, limit=5)
    print(vessel_calls)
    
    port_traffic = get_port_traffic("Barcelona", "2023-01-01", "2023-12-31")
    print(port_traffic)
    # Puedes agregar más ejemplos de uso aquí
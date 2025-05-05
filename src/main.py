"""
Script principal para el análisis y visualización de tráfico marítimo.
"""
import os
import argparse
import json
from datetime import datetime, timedelta

# Intentar importar pandas, pero seguir adelante si no está instalado
try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    print("AVISO: pandas no está instalado. Algunas funcionalidades podrían estar limitadas.")

try:
    from database import query_to_dataframe, get_vessel_info, get_vessel_port_calls, get_port_traffic
    from visualization import (plot_vessel_routes, plot_port_traffic_heatmap, 
                            plot_vessel_type_distribution, plot_monthly_port_calls,
                            save_map, save_figure)
    from analysis import (calculate_port_statistics, analyze_vessel_patterns,
                        compare_ports, analyze_regional_traffic,
                        get_vessel_types_distribution)
except ImportError as e:
    print(f"AVISO: Error al importar módulos: {e}")
    print("Es posible que algunas dependencias no estén instaladas.")
    print("Intente ejecutar: pip install -r requirements.txt")


def setup_argparse():
    """
    Configura el parser de argumentos de línea de comandos.
    
    Returns:
        argparse.ArgumentParser: Parser configurado
    """
    parser = argparse.ArgumentParser(description='Análisis de tráfico marítimo')
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: vessel-info
    vessel_info_parser = subparsers.add_parser('vessel-info', help='Obtener información de un buque')
    vessel_info_parser.add_argument('imo', type=int, help='Número IMO del buque')
    
    # Comando: vessel-pattern
    vessel_pattern_parser = subparsers.add_parser('vessel-pattern', 
                                               help='Analizar patrones de un buque')
    vessel_pattern_parser.add_argument('imo', type=int, help='Número IMO del buque')
    vessel_pattern_parser.add_argument('--days', type=int, default=365, 
                                     help='Número de días a analizar (default: 365)')
    vessel_pattern_parser.add_argument('--output', type=str, default='json',
                                     choices=['json', 'csv'],
                                     help='Formato de salida (default: json)')
    
    # Comando: port-stats
    port_stats_parser = subparsers.add_parser('port-stats', 
                                            help='Calcular estadísticas de un puerto')
    port_stats_parser.add_argument('port', type=str, help='Nombre del puerto')
    port_stats_parser.add_argument('--start-date', type=str, 
                                 help='Fecha de inicio (YYYY-MM-DD)')
    port_stats_parser.add_argument('--end-date', type=str, 
                                 help='Fecha de fin (YYYY-MM-DD)')
    port_stats_parser.add_argument('--output', type=str, default='json',
                                 choices=['json', 'csv'],
                                 help='Formato de salida (default: json)')
    
    # Comando: compare-ports
    compare_ports_parser = subparsers.add_parser('compare-ports',
                                              help='Comparar estadísticas de puertos')
    compare_ports_parser.add_argument('ports', type=str, nargs='+',
                                    help='Lista de puertos a comparar')
    compare_ports_parser.add_argument('--days', type=int, default=180,
                                    help='Número de días a analizar (default: 180)')
    compare_ports_parser.add_argument('--output', type=str, default='csv',
                                    choices=['json', 'csv'],
                                    help='Formato de salida (default: csv)')
    
    # Comando: region-analysis
    region_parser = subparsers.add_parser('region-analysis',
                                       help='Analizar tráfico de una región')
    region_parser.add_argument('country', type=str, help='Nombre del país o región')
    region_parser.add_argument('--start-date', type=str, 
                             help='Fecha de inicio (YYYY-MM-DD)')
    region_parser.add_argument('--end-date', type=str, 
                             help='Fecha de fin (YYYY-MM-DD)')
    region_parser.add_argument('--output', type=str, default='json',
                             choices=['json', 'csv'],
                             help='Formato de salida (default: json)')
    
    # Comando: port-traffic-vis
    port_vis_parser = subparsers.add_parser('port-traffic-vis',
                                         help='Visualizar tráfico en un puerto')
    port_vis_parser.add_argument('port', type=str, help='Nombre del puerto')
    port_vis_parser.add_argument('--days', type=int, default=90,
                               help='Número de días a visualizar (default: 90)')
    port_vis_parser.add_argument('--output', type=str, default='html',
                               choices=['html', 'png'],
                               help='Formato de salida (default: html)')
    
    # Comando: vessel-types
    vessel_types_parser = subparsers.add_parser('vessel-types',
                                             help='Analizar distribución de tipos de buques')
    vessel_types_parser.add_argument('--output', type=str, default='png',
                                   choices=['png', 'csv', 'json'],
                                   help='Formato de salida (default: png)')
    
    return parser


def output_result(result, output_format, filename_base):
    """
    Guarda el resultado en el formato especificado.
    
    Args:
        result: Resultado a guardar (dict, DataFrame, figure, map)
        output_format (str): Formato de salida ('json', 'csv', 'png', 'html')
        filename_base (str): Base para el nombre del archivo
    """
    # Obtener la ruta absoluta del directorio del proyecto
    # Asumimos que estamos en el directorio 'src'
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, 'data', 'results')
    
    # Crear directorio de resultados si no existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Generar el nombre de archivo con fecha y hora
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(results_dir, f"{filename_base}_{timestamp}")
    
    # Guardar según el formato
    if output_format == 'json' and isinstance(result, (dict, list)):
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Resultado guardado en: {filename}.json")
    
    elif output_format == 'csv':
        # Verificar si pandas está disponible
        if not pandas_available:
            print("ERROR: No se puede guardar en formato CSV porque pandas no está instalado.")
            print("Intente instalar pandas con: pip install pandas")
            # Guardar en JSON como alternativa
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Resultado guardado en formato JSON como alternativa: {filename}.json")
        else:
            # Si es un DataFrame, guardarlo como CSV
            if isinstance(result, pd.DataFrame):
                result.to_csv(f"{filename}.csv", index=False)
                print(f"Resultado guardado en: {filename}.csv")
            else:
                print("ERROR: El resultado no es un DataFrame, no se puede guardar como CSV.")
                # Guardar en JSON como alternativa
                with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Resultado guardado en formato JSON como alternativa: {filename}.json")
    
    elif output_format == 'png' and hasattr(result, 'savefig'):
        try:
            save_figure(result, filename)
            print(f"Gráfico guardado en: {filename}.png")
        except Exception as e:
            print(f"ERROR al guardar el gráfico: {e}")
    
    elif output_format == 'html' and hasattr(result, 'save'):
        try:
            save_map(result, filename)
            print(f"Mapa guardado en: {filename}.html")
        except Exception as e:
            print(f"ERROR al guardar el mapa: {e}")
    
    else:
        print(f"AVISO: No se pudo guardar el resultado en formato {output_format}.")
        print("El formato no es compatible con el tipo de resultado o las dependencias necesarias no están instaladas.")


def main():
    """
    Función principal del programa.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Si no se especifica un comando, mostrar ayuda
    if not args.command:
        parser.print_help()
        return
    
    # Procesamiento según el comando
    if args.command == 'vessel-info':
        if 'get_vessel_info' not in globals():
            print("ERROR: La función get_vessel_info no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
        result = get_vessel_info(args.imo)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'vessel-pattern':
        if 'analyze_vessel_patterns' not in globals():
            print("ERROR: La función analyze_vessel_patterns no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
        result = analyze_vessel_patterns(args.imo, args.days)
        output_result(result, args.output, f"vessel_pattern_{args.imo}")
    
    elif args.command == 'port-stats':
        if 'calculate_port_statistics' not in globals():
            print("ERROR: La función calculate_port_statistics no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
        result = calculate_port_statistics(args.port, args.start_date, args.end_date)
        output_result(result, args.output, f"port_stats_{args.port}")
    
    elif args.command == 'compare-ports':
        if 'compare_ports' not in globals():
            print("ERROR: La función compare_ports no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
        result = compare_ports(args.ports, args.days)
        output_result(result, args.output, "ports_comparison")
    
    elif args.command == 'region-analysis':
        if 'analyze_regional_traffic' not in globals():
            print("ERROR: La función analyze_regional_traffic no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
        result = analyze_regional_traffic(args.country, args.start_date, args.end_date)
        output_result(result, args.output, f"region_{args.country}")
    
    elif args.command == 'port-traffic-vis':
        if 'get_port_traffic' not in globals():
            print("ERROR: La función get_port_traffic no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
            
        # Calcular fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Obtener datos de tráfico
        traffic_data = get_port_traffic(
            args.port, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            limit=500
        )
        
        if args.output == 'html':
            if 'plot_port_traffic_heatmap' not in globals():
                print("ERROR: La función plot_port_traffic_heatmap no está disponible debido a errores de importación.")
                print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
                return
            # Crear mapa de tráfico
            map_result = plot_port_traffic_heatmap(traffic_data)
            output_result(map_result, 'html', f"port_traffic_{args.port}")
        else:
            if 'plot_monthly_port_calls' not in globals():
                print("ERROR: La función plot_monthly_port_calls no está disponible debido a errores de importación.")
                print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
                return
            # Crear gráfico mensual
            fig = plot_monthly_port_calls(traffic_data, args.port)
            output_result(fig, 'png', f"port_monthly_{args.port}")
    
    elif args.command == 'vessel-types':
        if 'get_vessel_types_distribution' not in globals():
            print("ERROR: La función get_vessel_types_distribution no está disponible debido a errores de importación.")
            print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
            return
            
        # Obtener distribución de tipos de buques
        vessel_types = get_vessel_types_distribution()
        
        if args.output == 'png':
            if 'plot_vessel_type_distribution' not in globals():
                print("ERROR: La función plot_vessel_type_distribution no está disponible debido a errores de importación.")
                print("Asegúrese de activar el entorno virtual con: source env/bin/activate")
                return
            # Crear gráfico
            fig = plot_vessel_type_distribution(vessel_types)
            output_result(fig, 'png', "vessel_types_distribution")
        else:
            # Guardar datos
            output_result(vessel_types, args.output, "vessel_types_distribution")


if __name__ == "__main__":
    main()

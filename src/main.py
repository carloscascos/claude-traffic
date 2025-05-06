"""
Script principal para el análisis y visualización de tráfico marítimo.
"""
import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
import json

from database import query_to_dataframe, get_vessel_info, get_vessel_port_calls, get_port_traffic
from visualization import (plot_vessel_routes, plot_port_traffic_heatmap, 
                          plot_vessel_type_distribution, plot_monthly_port_calls,
                          save_map, save_figure)
from analysis import (calculate_port_statistics, analyze_vessel_patterns,
                     compare_ports, analyze_regional_traffic,
                     get_vessel_types_distribution)


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
    # Crear directorio de resultados si no existe
    os.makedirs('../data/results', exist_ok=True)
    
    filename = f"../data/results/{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if output_format == 'json' and isinstance(result, (dict, list)):
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Resultado guardado en: {filename}.json")
    
    elif output_format == 'csv' and isinstance(result, pd.DataFrame):
        result.to_csv(f"{filename}.csv", index=False)
        print(f"Resultado guardado en: {filename}.csv")
    
    elif output_format == 'png' and hasattr(result, 'savefig'):
        save_figure(result, filename)
        print(f"Gráfico guardado en: {filename}.png")
    
    elif output_format == 'html' and hasattr(result, 'save'):
        save_map(result, filename)
        print(f"Mapa guardado en: {filename}.html")


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
        result = get_vessel_info(args.imo)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'vessel-pattern':
        result = analyze_vessel_patterns(args.imo, args.days)
        output_result(result, args.output, f"vessel_pattern_{args.imo}")
    
    elif args.command == 'port-stats':
        result = calculate_port_statistics(args.port, args.start_date, args.end_date)
        output_result(result, args.output, f"port_stats_{args.port}")
    
    elif args.command == 'compare-ports':
        result = compare_ports(args.ports, args.days)
        output_result(result, args.output, "ports_comparison")
    
    elif args.command == 'region-analysis':
        result = analyze_regional_traffic(args.country, args.start_date, args.end_date)
        output_result(result, args.output, f"region_{args.country}")
    
    elif args.command == 'port-traffic-vis':
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
            # Crear mapa de tráfico
            map_result = plot_port_traffic_heatmap(traffic_data)
            output_result(map_result, 'html', f"port_traffic_{args.port}")
        else:
            # Crear gráfico mensual
            fig = plot_monthly_port_calls(traffic_data, args.port)
            output_result(fig, 'png', f"port_monthly_{args.port}")
    
    elif args.command == 'vessel-types':
        # Obtener distribución de tipos de buques
        vessel_types = get_vessel_types_distribution()
        
        if args.output == 'png':
            # Crear gráfico
            fig = plot_vessel_type_distribution(vessel_types)
            output_result(fig, 'png', "vessel_types_distribution")
        else:
            # Guardar datos
            output_result(vessel_types, args.output, "vessel_types_distribution")


if __name__ == "__main__":
    main()

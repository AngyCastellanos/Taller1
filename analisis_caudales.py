"""
Análisis de Caudales - Estación La Victoria (Cundinamarca)
-----------------------------------------------------------
Este script analiza los datos de caudales de la estación La Victoria, calculando
estadísticos y generando visualizaciones como series temporales, boxplots,
histogramas y gráficos de frecuencias acumuladas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
from matplotlib.cm import get_cmap

# Configuración general de matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Definir una paleta de colores única para cada gráfico
# Usaremos combinaciones de colores vibrantes y distinguibles
COLORES = {
    'serie_temporal': '#FF5733',  # Rojo-naranja
    'boxplot_mensual': sns.color_palette("viridis", 12),
    'boxplot_anual': sns.color_palette("plasma", 25),
    'boxplot_estacional': sns.color_palette("magma", 4),
    'histograma': '#8E44AD',  # Morado
    'frecuencias_acum': '#3498DB',  # Azul
    'ajuste_normal': '#16A085',  # Verde-azulado
    'qq_plot': '#D35400',  # Naranja oscuro
    'frecuencias_abs': '#00B0F0',  # Azul claro
    'frecuencias_rel': '#92D050',  # Verde claro
    'dist_normal': '#9B59B6',  # Morado claro
    'frecuencias_acum_normal': '#E74C3C'  # Rojo
}

def graficar_frecuencias_absolutas(datos):
    """Genera histograma de frecuencias absolutas."""
    plt.figure(figsize=(10, 6))
    
    # Definir los bins
    bins = np.linspace(min(datos), max(datos), 16)
    
    # Calcular el histograma
    hist, bin_edges = np.histogram(datos, bins=bins)
    
    # Crear las barras con el color específico
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', 
            color=COLORES['frecuencias_abs'], edgecolor='black')
    
    # Añadir valores sobre las barras
    for i, v in enumerate(hist):
        plt.text(bin_edges[i] + np.diff(bin_edges)[0]/2, v + 0.5, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    # Configurar el gráfico
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Frecuencia absoluta', fontweight='bold')
    plt.title('Histograma de Frecuencias Absolutas', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('frecuencias_absolutas.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_frecuencias_relativas(datos):
    """Genera histograma de frecuencias relativas."""
    plt.figure(figsize=(10, 6))
    
    # Definir los bins
    bins = np.linspace(min(datos), max(datos), 16)
    
    # Calcular el histograma
    hist, bin_edges = np.histogram(datos, bins=bins)
    freq_rel = hist / len(datos)
    
    # Crear las barras con el color específico
    plt.bar(bin_edges[:-1], freq_rel, width=np.diff(bin_edges), align='edge',
            color=COLORES['frecuencias_rel'], edgecolor='black')
    
    # Añadir valores sobre las barras
    for i, v in enumerate(freq_rel):
        plt.text(bin_edges[i] + np.diff(bin_edges)[0]/2, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Configurar el gráfico
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Frecuencia relativa', fontweight='bold')
    plt.title('Histograma de Frecuencias Relativas', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('frecuencias_relativas.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_distribucion_normal(datos):
    """Genera gráfico de distribución normal con frecuencias relativas."""
    plt.figure(figsize=(10, 6))
    
    # Calcular parámetros de la distribución normal
    mu, sigma = stats.norm.fit(datos)
    
    # Crear puntos para la curva normal
    x = np.linspace(min(datos) - sigma, max(datos) + sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    
    # Graficar la curva normal
    plt.plot(x, y, color=COLORES['dist_normal'], linewidth=2, label='Distribución normal')
    
    # Calcular y graficar frecuencias relativas
    bins = np.linspace(min(datos), max(datos), 16)
    hist, bin_edges = np.histogram(datos, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, hist, 'ro-', linewidth=1, markersize=4,
             label='Frecuencia relativa')
    
    # Configurar el gráfico
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Densidad de probabilidad', fontweight='bold')
    plt.title('Distribución Normal Estándar', fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir eje secundario
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Frecuencia relativa', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('distribucion_normal.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_frecuencias_acumuladas(datos):
    """Genera gráfico de frecuencias acumuladas."""
    plt.figure(figsize=(10, 6))
    
    # Calcular frecuencias acumuladas
    datos_ordenados = np.sort(datos)
    freq_acum = np.arange(1, len(datos) + 1) / len(datos)
    
    # Graficar frecuencias acumuladas
    plt.plot(datos_ordenados, freq_acum * 100, color=COLORES['frecuencias_acum_normal'], linewidth=2)
    plt.fill_between(datos_ordenados, freq_acum * 100, alpha=0.3, color=COLORES['frecuencias_acum_normal'])
    
    # Configurar el gráfico
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Frecuencia acumulada (%)', fontweight='bold')
    plt.title('Distribución Normal Estándar, Función Acumulada', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    
    # Añadir líneas de referencia
    plt.axvline(x=np.median(datos), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=50, color='blue', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('frecuencias_acumuladas.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_serie_temporal(df):
    """Genera un gráfico de serie temporal de los caudales."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['FECHA'], df['CAUDAL'], color=COLORES['serie_temporal'], linewidth=1.5, marker='o', markersize=4)
    plt.xlabel('Fecha', fontweight='bold')
    plt.ylabel('Caudal (m³/s)', fontweight='bold')
    plt.title('Serie Temporal de Caudales - Estación La Victoria (1997-2021)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, df['CAUDAL'].max() * 1.1)
    plt.savefig('serie_temporal_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis de la serie temporal
    print("\nAnálisis de la Serie Temporal:")
    tendencia = "ascendente" if df['CAUDAL'].diff().mean() > 0 else "descendente"
    print(f"- La serie de caudales muestra una tendencia general {tendencia} de {abs(df['CAUDAL'].diff().mean()):.5f} m³/s por día.")
    
    # Identificar valores extremos
    umbral_alto = df['CAUDAL'].mean() + 2 * df['CAUDAL'].std()
    extremos = df[df['CAUDAL'] > umbral_alto]
    
    if not extremos.empty:
        print("- Se detectaron los siguientes eventos extremos de caudal:")
        for _, row in extremos.iterrows():
            print(f"  * {row['FECHA'].strftime('%B %Y')}: {row['CAUDAL']:.3f} m³/s")
    else:
        print("- No se detectaron eventos extremos significativos de caudal.")

def graficar_boxplots(df_monthly, df_annual, df_seasonal):
    """Genera boxplots a diferentes resoluciones temporales."""
    # 1. Boxplot mensual
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MES', y='CAUDAL', data=df_monthly, palette=COLORES['boxplot_mensual'])
    plt.title('Distribución de Caudales por Mes - Estación La Victoria (1997-2021)', fontweight='bold')
    plt.xlabel('Mes', fontweight='bold')
    plt.ylabel('Caudal (m³/s)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('boxplot_mensual_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot anual
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='AÑO', y='CAUDAL', data=df_monthly, palette=COLORES['boxplot_anual'])
    plt.title('Distribución de Caudales por Año - Estación La Victoria (1997-2021)', fontweight='bold')
    plt.xlabel('Año', fontweight='bold')
    plt.ylabel('Caudal (m³/s)', fontweight='bold')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('boxplot_anual_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Boxplot estacional
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ESTACION', y='CAUDAL', data=df_seasonal, palette=COLORES['boxplot_estacional'],
                order=['DEF', 'MAM', 'JJA', 'SON'])
    plt.title('Distribución de Caudales por Estación - Estación La Victoria (1997-2021)', fontweight='bold')
    plt.xlabel('Estación', fontweight='bold')
    plt.ylabel('Caudal (m³/s)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('boxplot_estacional_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis de boxplots
    print("\nAnálisis de los Boxplots:")
    
    # Análisis mensual
    meses_altos = df_monthly.groupby('MES')['CAUDAL'].mean().sort_values(ascending=False).index[:3].tolist()
    meses_bajos = df_monthly.groupby('MES')['CAUDAL'].mean().sort_values().index[:3].tolist()
    
    print(f"- Los meses con mayores caudales promedio son: {', '.join(meses_altos)}")
    print(f"- Los meses con menores caudales promedio son: {', '.join(meses_bajos)}")
    
    # Análisis estacional
    estaciones_orden = df_seasonal.groupby('ESTACION')['CAUDAL'].mean().sort_values(ascending=False).index.tolist()
    print(f"- Estaciones ordenadas por caudal promedio (mayor a menor): {', '.join(estaciones_orden)}")
    
    # Análisis interanual
    anos_extremos = df_annual.sort_values('CAUDAL', ascending=False)
    print(f"- El año con mayor caudal promedio fue {anos_extremos.iloc[0]['AÑO']} ({anos_extremos.iloc[0]['CAUDAL']:.3f} m³/s)")
    print(f"- El año con menor caudal promedio fue {anos_extremos.iloc[-1]['AÑO']} ({anos_extremos.iloc[-1]['CAUDAL']:.3f} m³/s)")

def graficar_histograma_y_acumulado(datos):
    """Genera histograma de frecuencias y gráfico de frecuencias acumuladas."""
    
    # Calcular número de bins usando la regla de Sturges
    n_bins = int(np.ceil(1 + 3.322 * np.log10(len(datos))))
    
    # Calcular histograma
    conteos, bordes = np.histogram(datos, bins=n_bins)
    anchura_bin = bordes[1] - bordes[0]
    
    # Frecuencias relativas
    freq_relativa = conteos / len(datos)
    
    # Estimación de probabilidades por bin
    prob_bin = freq_relativa / anchura_bin
    
    # Frecuencias acumuladas
    freq_acumulada = np.cumsum(freq_relativa)
    
    # 1. Histograma de frecuencias absolutas y relativas
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Barras - frecuencias absolutas
    ax1.bar(bordes[:-1], conteos, width=anchura_bin, alpha=0.7, color=COLORES['histograma'], 
            edgecolor='black', label='Frecuencia Absoluta')
    ax1.set_xlabel('Caudal (m³/s)', fontweight='bold')
    ax1.set_ylabel('Frecuencia Absoluta', color='blue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Línea superpuesta - densidad de probabilidad
    ax1.plot(bordes[:-1] + anchura_bin/2, prob_bin, 'ro-', label='Densidad de Probabilidad')
    
    # Eje secundario para frecuencias relativas
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frecuencia Relativa', color='red', fontweight='bold')
    ax2.bar(bordes[:-1], freq_relativa, width=anchura_bin, alpha=0.3, color='salmon', 
            edgecolor='red', label='Frecuencia Relativa')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.title('Histograma de Caudales - Estación La Victoria (1997-2021)', fontweight='bold')
    fig.tight_layout()
    
    # Combinación de leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.savefig('histograma_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de frecuencias acumuladas
    plt.figure(figsize=(10, 6))
    plt.plot(bordes[:-1] + anchura_bin/2, freq_acumulada * 100, 'o-', lw=2, 
             color=COLORES['frecuencias_acum'])
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Frecuencia Acumulada (%)', fontweight='bold')
    plt.title('Frecuencias Acumuladas de Caudales - Estación La Victoria (1997-2021)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(np.arange(0, 101, 10))
    plt.tight_layout()
    plt.savefig('frecuencias_acumuladas_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis del histograma y frecuencias
    print("\nAnálisis del Histograma y Frecuencias:")
    
    # Rango de caudales más frecuentes (moda)
    bin_max = np.argmax(conteos)
    rango_modal = f"{bordes[bin_max]:.3f} - {bordes[bin_max+1]:.3f}"
    print(f"- El rango de caudales más frecuente es: {rango_modal} m³/s")
    print(f"  Este rango contiene {conteos[bin_max]} observaciones ({freq_relativa[bin_max]*100:.1f}% del total).")
    
    # Análisis de percentiles para una mejor interpretación
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    valores_percentiles = np.percentile(datos, percentiles)
    
    print("- Valores de caudal según percentiles:")
    for p, v in zip(percentiles, valores_percentiles):
        print(f"  * Percentil {p}%: {v:.3f} m³/s")
    
    # Interpretación de probabilidades
    print("\n- Interpretación de probabilidades basada en el histograma:")
    
    # Probabilidad de excedencia para algunos valores de referencia
    valores_ref = [0.1, 0.2, 0.5, 1.0]
    for val in valores_ref:
        prob = (datos > val).sum() / len(datos) * 100
        print(f"  * Probabilidad de que el caudal exceda {val} m³/s: {prob:.1f}%")

def ajuste_distribucion_normal(datos):
    """Ajusta los datos a una distribución normal y muestra el resultado."""
    
    # Parámetros de la distribución normal
    mu, sigma = stats.norm.fit(datos)
    
    # Generar la distribución normal ajustada
    x = np.linspace(min(datos), max(datos), 100)
    y = stats.norm.pdf(x, mu, sigma)
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    
    # Histograma normalizado
    plt.hist(datos, bins=15, density=True, alpha=0.6, color=COLORES['ajuste_normal'], edgecolor='black',
             label='Histograma de datos')
    
    # Curva de la distribución normal
    plt.plot(x, y, 'r-', linewidth=2, label=f'Normal: u={mu:.3f}, s={sigma:.3f}')
    
    # Añadir detalles
    plt.title('Ajuste a Distribución Normal - Caudales Estación La Victoria', fontweight='bold')
    plt.xlabel('Caudal (m³/s)', fontweight='bold')
    plt.ylabel('Densidad de Probabilidad', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('ajuste_normal_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test de normalidad (Shapiro-Wilk)
    stat, p = stats.shapiro(datos)
    
    # Histograma Q-Q plot para normalidad
    plt.figure(figsize=(10, 6))
    
    # Modificar el estilo del Q-Q plot para usar nuestros colores
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    res = stats.probplot(datos, dist="norm", plot=ax)
    
    # Personalizar el gráfico Q-Q
    ax.get_lines()[0].set_markerfacecolor(COLORES['qq_plot'])
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[1].set_color('black')
    
    plt.title('Q-Q Plot para Evaluación de Normalidad - Caudales Estación La Victoria', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('qq_plot_caudales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis del ajuste a la distribución normal
    print("\nAnálisis del Ajuste a la Distribución Normal:")
    print(f"- Parámetros de la distribución normal ajustada: Media (u)={mu:.4f}, Desviación estándar (s)={sigma:.4f}")
    
    # Interpretación del test de normalidad
    print(f"- Resultados del test de Shapiro-Wilk: estadístico={stat:.4f}, p-valor={p:.6f}")
    
    if p < 0.05:
        print("  El p-valor es menor que 0.05, lo que sugiere que los datos NO siguen una distribución normal.")
        
        # Sugerencias sobre distribuciones alternativas
        if stats.skew(datos) > 0.5:
            print("  Dada la asimetría positiva, se podría considerar ajustar a una distribución lognormal o gamma.")
        else:
            print("  Se podría considerar otras distribuciones como Weibull, exponencial o una distribución mixta.")
    else:
        print("  El p-valor es mayor que 0.05, lo que sugiere que los datos se pueden aproximar a una distribución normal.")
    
    print("\n- Interpretación del Q-Q Plot:")
    print("  El gráfico Q-Q muestra qué tan bien los datos se ajustan a una distribución normal.")
    print("  Los puntos que siguen la línea diagonal indican un buen ajuste a la normalidad,")
    print("  mientras que las desviaciones indican apartamientos de la normalidad.")
    
    # Cálculo de probabilidades utilizando la distribución normal ajustada
    print("\n- Probabilidades basadas en la distribución normal ajustada:")
    valores_ref = [0.1, 0.2, 0.5, 1.0]
    for val in valores_ref:
        prob = (1 - stats.norm.cdf(val, mu, sigma)) * 100
        print(f"  * Probabilidad teórica de que el caudal exceda {val} m³/s: {prob:.1f}%")

def main():
    # Cargar los datos
    print("Cargando datos de caudales...")
    df = pd.read_csv('caudales_victoria.csv')
    
    # Preprocesamiento y transformación de datos
    print("Preprocesando datos...")
    df_long = df.melt(id_vars=['AÑO'], 
                       value_vars=['ENERO', 'FEBRE', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 
                                  'JULIO', 'AGOST', 'SEPTI', 'OCTUB', 'NOVIE', 'DICIE'],
                       var_name='MES', value_name='CAUDAL')
    
    # Convertir a formato fecha para análisis temporales
    meses_num = {'ENERO': 1, 'FEBRE': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
                 'JULIO': 7, 'AGOST': 8, 'SEPTI': 9, 'OCTUB': 10, 'NOVIE': 11, 'DICIE': 12}
    df_long['MES_NUM'] = df_long['MES'].map(meses_num)
    df_long['FECHA'] = pd.to_datetime(df_long['AÑO'].astype(str) + '-' + df_long['MES_NUM'].astype(str) + '-01')
    
    # Ordenar por fecha
    df_long = df_long.sort_values('FECHA')
    
    # Crear dataframes a diferentes resoluciones temporales
    df_monthly = df_long.copy()
    df_annual = df_long.groupby('AÑO').agg({'CAUDAL': 'mean'}).reset_index()
    df_seasonal = df_long.copy()
    
    # Asignar estaciones (temporadas) del año
    def asignar_estacion(mes):
        if mes in [12, 1, 2]:  # Diciembre, Enero, Febrero
            return 'DEF'
        elif mes in [3, 4, 5]:  # Marzo, Abril, Mayo
            return 'MAM'
        elif mes in [6, 7, 8]:  # Junio, Julio, Agosto
            return 'JJA'
        else:  # Septiembre, Octubre, Noviembre
            return 'SON'
    
    df_seasonal['ESTACION'] = df_seasonal['MES_NUM'].apply(asignar_estacion)
    df_seasonal = df_seasonal.groupby(['AÑO', 'ESTACION']).agg({'CAUDAL': 'mean'}).reset_index()
    
    # Extracción de todos los valores de caudal
    caudal_valores = df_long['CAUDAL'].values
    
    # 1. Cálculo de estadísticos
    print("\n=== ANÁLISIS ESTADÍSTICO ===")
    calcular_estadisticos(caudal_valores)
    
    # 2. Gráficos
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    # Serie temporal
    graficar_serie_temporal(df_long)
    
    # Boxplots a diferentes resoluciones temporales
    graficar_boxplots(df_monthly, df_annual, df_seasonal)
    
    # Histograma y frecuencias acumuladas
    graficar_histograma_y_acumulado(caudal_valores)
    
    # Ajuste a distribución normal
    ajuste_distribucion_normal(caudal_valores)
    
    # Generar todas las gráficas de frecuencias
    print("\nGenerando gráficas de frecuencias...")
    graficar_frecuencias_absolutas(caudal_valores)
    graficar_frecuencias_relativas(caudal_valores)
    graficar_distribucion_normal(caudal_valores)
    graficar_frecuencias_acumuladas(caudal_valores)
    
    print("\nAnálisis completado. Todas las gráficas han sido guardadas.")

def calcular_estadisticos(datos):
    """Calcula y muestra los estadísticos descriptivos de los datos de caudal."""
    
    # Estadísticos básicos
    media = np.mean(datos)
    mediana = np.median(datos)
    # Calcular la moda (puede haber múltiples valores modales en datos continuos)
    moda = stats.mode(datos, keepdims=True).mode[0]
    desviacion_std = np.std(datos)
    varianza = np.var(datos)
    
    # Coeficiente de variación (CV)
    cv = (desviacion_std / media) * 100
    
    # Oblicuidad (skewness)
    oblicuidad = stats.skew(datos)
    
    # Mostrar resultados
    print(f"Media: {media:.4f} m³/s")
    print(f"Mediana: {mediana:.4f} m³/s")
    print(f"Moda (aproximada): {moda:.4f} m³/s")
    print(f"Desviación Estándar: {desviacion_std:.4f} m³/s")
    print(f"Varianza: {varianza:.4f} (m³/s)²")
    print(f"Coeficiente de Variación: {cv:.2f}%")
    print(f"Oblicuidad (Skewness): {oblicuidad:.4f}")
    
    # Interpretación de los estadísticos
    print("\nInterpretación:")
    print(f"- La media de caudal es {media:.4f} m³/s, lo que representa el valor promedio en el período analizado.")
    print(f"- La mediana de {mediana:.4f} m³/s indica el valor central de los datos.")
    
    if media > mediana:
        print("- Como la media es mayor que la mediana, la distribución muestra una asimetría positiva.")
    elif media < mediana:
        print("- Como la media es menor que la mediana, la distribución muestra una asimetría negativa.")
    else:
        print("- La media y mediana son similares, sugiriendo una distribución aproximadamente simétrica.")
    
    print(f"- El coeficiente de variación de {cv:.2f}% indica el grado de dispersión relativa de los datos.")
    
    if cv < 30:
        print("  Este CV es relativamente bajo, lo que indica una variabilidad moderada en los caudales.")
    else:
        print("  Este CV es alto, lo que refleja una alta variabilidad en los caudales.")
    
    if oblicuidad > 0.5:
        print(f"- La oblicuidad positiva ({oblicuidad:.4f}) indica una cola más larga hacia valores altos,")
        print("  sugiriendo la presencia de eventos extremos de alto caudal.")
    elif oblicuidad < -0.5:
        print(f"- La oblicuidad negativa ({oblicuidad:.4f}) indica una cola más larga hacia valores bajos.")
    else:
        print(f"- La oblicuidad cercana a cero ({oblicuidad:.4f}) sugiere una distribución aproximadamente simétrica.")

if __name__ == "__main__":
    main() 
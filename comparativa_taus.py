#Comparador tau y resultados de NE@citrato en distintos congelamientos 
#%% Librerias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% plot_ciclos_promedio
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)

    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('Campo magnético (kA/m)')
    plt.ylabel('Magnetización (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()
#%% lector_resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value

    # Leer los datos del archivo
    data = pd.read_table(path, header=17,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% Lector_ciclos
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Rsultados 250716_NE-citrato_250609_c_paper_descongelamiento
res_csC=glob(os.path.join('../250716_NE@citrato_250609_c_paper_descongelamiento','cong_sin_campo', '**', '*resultados.txt'),recursive=True)
res_csC.sort()
meta_csC_0, _,time_csC_0,T_csC_0,_,_,_,_,_,_,_,_,SAR_csC_0,tau_csC_0,_=lector_resultados(res_csC[0])
meta_csC_1, _,time_csC_1,T_csC_1,_,_,_,_,_,_,_,_,SAR_csC_1,tau_csC_1,_=lector_resultados(res_csC[1])
meta_csC_2, _,time_csC_2,T_csC_2,_,_,_,_,_,_,_,_,SAR_csC_2,tau_csC_2,_=lector_resultados(res_csC[2])

res_cCT=glob(os.path.join('../250716_NE@citrato_250609_c_paper_descongelamiento','cong_con_campo', '**', '*resultados.txt'),recursive=True)
res_cCT.sort()
meta_cCT_0, _,time_cCT_0,T_cCT_0,_,_,_,_,_,_,_,_,SAR_cCT_0,tau_cCT_0,_=lector_resultados(res_cCT[0])
meta_cCT_1, _,time_cCT_1,T_cCT_1,_,_,_,_,_,_,_,_,SAR_cCT_1,tau_cCT_1,_=lector_resultados(res_cCT[1])
meta_cCT_2, _,time_cCT_2,T_cCT_2,_,_,_,_,_,_,_,_,SAR_cCT_2,tau_cCT_2,_=lector_resultados(res_cCT[2])

res_cCA=glob(os.path.join('38', '**', '*resultados.txt'),recursive=True)
res_cCA.sort()
meta_cCA_0, _,time_cCA_0,T_cCA_0,_,_,_,_,_,_,_,_,SAR_cCA_0,tau_cCA_0,_=lector_resultados(res_cCA[0])
meta_cCA_1, _,time_cCA_1,T_cCA_1,_,_,_,_,_,_,_,_,SAR_cCA_1,tau_cCA_1,_=lector_resultados(res_cCA[1])
meta_cCA_2, _,time_cCA_2,T_cCA_2,_,_,_,_,_,_,_,_,SAR_cCA_2,tau_cCA_2,_=lector_resultados(res_cCA[2])

# %% PLOT TAUS ALL
fig, (a,b,c)=plt.subplots(nrows=3,constrained_layout=True, sharex=True, sharey=True,figsize=(9,7))

a.set_title('cong s/ campo',loc='left')
#a.plot(T_csC_0,tau_csC_0,'.-')
a.plot(T_csC_1,tau_csC_1,'.-')
a.plot(T_csC_2,tau_csC_2,'.-')

b.set_title('cong c/ campo transversal',loc='left')
b.plot(T_cCT_0,tau_cCT_0,'.-')
b.plot(T_cCT_1,tau_cCT_1,'.-')
b.plot(T_cCT_2,tau_cCT_2,'.-')

c.set_title('cong c/ campo axial',loc='left')
c.plot(T_cCA_0,tau_cCA_0,'.-')
c.plot(T_cCA_1,tau_cCA_1,'.-')
c.plot(T_cCA_2,tau_cCA_2,'.-')

plt.xlim(-20,20)
plt.suptitle('NE@citrato 25009_c en descongelamiento  -  135 kHz/38 kA/m')
for ax in (a,b,c):
    ax.grid()
    ax.set_ylabel(r'$\tau$ (ns)')
c.set_xlabel('Temperatura (°C)')
plt.savefig('comparativa_taus.png',dpi=300)
#%%
#%% promediar_tau
def promediar_tau(T_list, tau_list, nombre_conjunto, temp_min=-20, temp_max=20, intervalo_temp=1):
    """
    Parámetros:
    - T_list: Lista de arrays de temperatura (ej: [T_csC_0, T_csC_1, T_csC_2])
    - tau_list: Lista de arrays de tau correspondientes (ej: [tau_csC_0, tau_csC_1, tau_csC_2])
    - nombre_conjunto: Nombre del conjunto (ej: "csC", "cCT" o "cCA")
    - temp_min: Temperatura mínima para el análisis (default: -20)
    - temp_max: Temperatura máxima para el análisis (default: 20)
    - intervalo_temp: Ancho del intervalo de temperatura (default: 1°C)
    
    Retorna:
    - T_intervalos: Array con los límites inferiores de los intervalos de temperatura
    - prom_tau: Array con los promedios de tau (sin NaN)
    - err_tau: Array con los errores estándar (sin NaN)
    - err_temperatura: Array con el error de temperatura (intervalo_temp/2)
    - counts: Array con el número de datos en cada intervalo
    """
    # Recortar datos entre temp_min y temp_max °C y concatenar
    T_total = np.array([])
    tau_total = np.array([])
    
    for T, tau in zip(T_list, tau_list):
        idx = np.nonzero((T >= temp_min) & (T <= temp_max))
        T_total = np.concatenate((T_total, T[idx]))
        tau_total = np.concatenate((tau_total, tau[idx]))
    
    # Crear intervalos de temperatura (desde temp_min hasta temp_max inclusive)
    T_intervalos = np.arange(temp_min, temp_max + intervalo_temp, intervalo_temp)
    
    # Calcular promedios, errores y conteos (ignorando NaN)
    prom_tau = []
    err_tau = []
    counts = []
    
    for temp in T_intervalos:
        mask = (T_total >= temp) & (T_total < temp + intervalo_temp)
        tau_intervalo = tau_total[mask]
        
        # Calcular media, std y conteo ignorando NaN
        mean_val = np.nanmean(tau_intervalo)
        std_val = np.nanstd(tau_intervalo)
        count = np.sum(~np.isnan(tau_intervalo))
        
        prom_tau.append(mean_val)
        err_tau.append(std_val)
        counts.append(count)
    
    # Convertir a arrays de numpy y eliminar intervalos con NaN
    prom_tau = np.array(prom_tau)
    err_tau = np.array(err_tau)
    counts = np.array(counts)
    
    # Máscara para valores no NaN
    mask_no_nan = ~np.isnan(prom_tau)
    
    # Filtrar resultados
    T_intervalos = T_intervalos[mask_no_nan]
    prom_tau = prom_tau[mask_no_nan]
    err_tau = err_tau[mask_no_nan]
    counts = counts[mask_no_nan]
    
    # Error de temperatura (mitad del intervalo)
    err_temperatura = np.full(len(T_intervalos), intervalo_temp/2)
    
    # Mostrar resultados con columna de conteo
    print(f"\nResultados para {nombre_conjunto} ({temp_min}°C a {temp_max}°C):")
    print("Intervalo de Temp  |   N   |   Promedio de Tau   |   Error Std")
    print("----------------------------------------------------------------------")
    for i, temp in enumerate(T_intervalos):
        print(f"{temp:5.2f} to {temp + intervalo_temp:5.2f} °C | {counts[i]:4d} | {prom_tau[i]:15.2e} | {err_tau[i]:12.2e}")
    
    return T_intervalos, prom_tau, err_tau, err_temperatura
#%% Ejecucion
T_csC, prom_tau_csC, err_tau_csC, err_temp_csC = promediar_tau([ T_csC_1, T_csC_2], [tau_csC_1, tau_csC_2], "csC")
T_cCT, prom_tau_cCT, err_tau_cCT, err_temp_cCT = promediar_tau([T_cCT_0, T_cCT_1, T_cCT_2], [tau_cCT_0, tau_cCT_1, tau_cCT_2], "cCT")
T_cCA, prom_tau_cCA, err_tau_cCA, err_temp_cCA = promediar_tau([T_cCA_0, T_cCA_1, T_cCA_2], [tau_cCA_0, tau_cCA_1, tau_cCA_2], "cCA")

#%% PLOT TAU
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.set_title(r'$\tau$')
ax.set_ylabel(r'$\tau$ (s)')

# Graficar con errores (xerr = err_temperatura)
ax.errorbar(x=T_csC, y=prom_tau_csC, xerr=err_temp_csC, yerr=err_tau_csC, capsize=4, fmt='.-', label='cong s/ Campo')
ax.errorbar(x=T_cCT, y=prom_tau_cCT, xerr=err_temp_cCT, yerr=err_tau_cCT, capsize=4, fmt='.-', label='cong c/Campo Trasversal')
ax.errorbar(x=T_cCA, y=prom_tau_cCA, xerr=err_temp_cCA, yerr=err_tau_cCA, capsize=4, fmt='.-', label='cong c/Campo Axial')

ax.set_xlabel('Temperature (°C)')
plt.xlim(-21, 21)
ax.grid()
ax.legend()
plt.savefig('comparativa_promedio_taus.png',dpi=300)
plt.show()

#%% PLOT temperatura vs tiempo 
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, NullFormatter

# Configurar el normalizador de colores para la temperatura
temp_min = min(np.nanmin(T_csC_1), np.nanmin(T_csC_2), 
               np.nanmin(T_cCT_0), np.nanmin(T_cCT_1), np.nanmin(T_cCT_2),
               np.nanmin(T_cCA_0), np.nanmin(T_cCA_1), np.nanmin(T_cCA_2))
temp_max = max(np.nanmax(T_csC_1), np.nanmax(T_csC_2),
              np.nanmax(T_cCT_0), np.nanmax(T_cCT_1), np.nanmax(T_cCT_2),
              np.nanmax(T_cCA_0), np.nanmax(T_cCA_1), np.nanmax(T_cCA_2))
norm = Normalize(vmin=temp_min, vmax=temp_max)


# Ajustar tiempos para que empiecen en 0
time_csC_1 = time_csC_1 - time_csC_1[0]
time_csC_2 = time_csC_2 - time_csC_2[0]
time_cCT_0 = time_cCT_0 - time_cCT_0[0]
time_cCT_1 = time_cCT_1 - time_cCT_1[0]
time_cCT_2 = time_cCT_2 - time_cCT_2[0]
time_cCA_0 = time_cCA_0 - time_cCA_0[0]
time_cCA_1 = time_cCA_1 - time_cCA_1[0]
time_cCA_2 = time_cCA_2 - time_cCA_2[0]
#%% Filtrar los datos para que solo incluyan temperaturas entre -20 y 20 °C

def filtrar_por_temperatura(tiempo, temperatura, tmin=-20, tmax=20):
    mask = (temperatura >= tmin) & (temperatura <= tmax)
    return tiempo[mask], temperatura[mask]

# Congelado sin campo
time_csC_1, T_csC_1 = filtrar_por_temperatura(time_csC_1, T_csC_1)
time_csC_2, T_csC_2 = filtrar_por_temperatura(time_csC_2, T_csC_2)

# Congelado con campo transversal
time_cCT_0, T_cCT_0 = filtrar_por_temperatura(time_cCT_0, T_cCT_0)
time_cCT_1, T_cCT_1 = filtrar_por_temperatura(time_cCT_1, T_cCT_1)
time_cCT_2, T_cCT_2 = filtrar_por_temperatura(time_cCT_2, T_cCT_2)

# Congelado con campo axial
time_cCA_0, T_cCA_0 = filtrar_por_temperatura(time_cCA_0, T_cCA_0)
time_cCA_1, T_cCA_1 = filtrar_por_temperatura(time_cCA_1, T_cCA_1)
time_cCA_2, T_cCA_2 = filtrar_por_temperatura(time_cCA_2, T_cCA_2)

#%% Plot Congelado sin campo (solo muestras 1 y 2 como en el análisis anterior)
fig, (a, b, c) = plt.subplots(nrows=3, constrained_layout=True, sharex=True, sharey=True, figsize=(10, 8))
a.set_title(' Congelado sin campo', loc='left',y=0.89)
sc1 = a.scatter(time_csC_1/60, T_csC_1, c=T_csC_1, cmap=viridis, norm=norm, label='Muestra 1')
sc2 = a.scatter(time_csC_2/60, T_csC_2, c=T_csC_2, cmap=viridis, norm=norm, label='Muestra 2')
a.set_ylabel('Temperatura (°C)')

# Congelado con campo transversal
b.set_title(' Congelado con campo transversal', loc='left', y=0.89)
b.scatter(time_cCT_0/60, T_cCT_0, c=T_cCT_0, cmap=viridis, norm=norm, label='Muestra 0')
b.scatter(time_cCT_1/60, T_cCT_1, c=T_cCT_1, cmap=viridis, norm=norm, label='Muestra 1')
b.scatter(time_cCT_2/60, T_cCT_2, c=T_cCT_2, cmap=viridis, norm=norm, label='Muestra 2')
b.set_ylabel('Temperatura (°C)')

# Congelado con campo axial
c.set_title(' Congelado con campo axial', loc='left', y=0.89)
c.scatter(time_cCA_0/60, T_cCA_0, c=T_cCA_0, cmap=viridis, norm=norm, label='Muestra 0')
c.scatter(time_cCA_1/60, T_cCA_1, c=T_cCA_1, cmap=viridis, norm=norm, label='Muestra 1')
c.scatter(time_cCA_2/60, T_cCA_2, c=T_cCA_2, cmap=viridis, norm=norm, label='Muestra 2')
c.set_ylabel('Temperatura (°C)')
c.set_xlabel('Tiempo (minutos)')

for ax in (a, b, c):
    ax.grid(which='major')
    # Eje y: cada 5 grados
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    # Eje x: cada 20 segundos (20/60 minutos)
    ax.xaxis.set_minor_locator(MultipleLocator(20/60))
    ax.grid(which='minor', linestyle=':', linewidth=0.7, alpha=0.6)
    # Quitar los números de los ticks menores
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

# Barra de color común
cbar = fig.colorbar(sc1, ax=[a, b, c], orientation='vertical', label='Temperatura (°C)')

# Ajustes globales
plt.suptitle('Temperatura durante el descongelamiento', y=1.02)

#plt.savefig('evolucion_temperatura_viridis_tiempo_ajustado.png', dpi=300, bbox_inches='tight')
plt.show()
# %%

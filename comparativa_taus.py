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
#%%
#LECTOR CICLOS
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
#%% Ploteo todos los ciclos 
plot_ciclos_promedio('ex_citrato')
plot_ciclos_promedio('ex_hierro')
plot_ciclos_promedio('in_situ')
#%% Comparo a mismos idc/campo  
idcs = [150,135,120,105,90,75,60,45,30]
for idc in idcs:
    ciclos = glob(os.path.join('**/*2025*','*'+str(idc)+'dA'+'*ciclo_promedio*'),recursive=True)

    _,_,_,H_citrato,M_citrato,meta_citrato = lector_ciclos(ciclos[0])
    _,_,_,H_hierro,M_hierro,meta_hierro = lector_ciclos(ciclos[1])
    _,_,_,H_insitu,M_insitu,meta_insitu = lector_ciclos(ciclos[2])
    H_max = (idc/10*float(meta_citrato['pendiente_HvsI '])+float(meta_citrato['ordenada_HvsI ']))/1000
    frec = meta_citrato['frecuencia']/1000
    titulo=f'{H_max:.1f} kA/m - {frec:.1f} kHz'


    fig, ax = plt.subplots(nrows=1,figsize=(6,5),constrained_layout=True)
    ax.plot(H_insitu/1000,M_insitu,c='tab:red',label='in situ')
    ax.plot(H_citrato/1000,M_citrato,c='tab:green',label='ex citrato')
    ax.plot(H_hierro/1000,M_hierro,c='tab:blue',label='ex hierro')
    ax.grid()
    ax.set_xlabel('H (kA/m)')
    ax.set_ylabel('M (A/m)')
    ax.set_title(titulo,fontsize=12)
    ax.legend(title='Ferrogel',ncol=1)
    # ax.set_xlim(0,60e3)
    # ax.set_ylim(0,)
    plt.savefig('comparativa_HM_tancredi_'+str(idc)+'.png',dpi=400)
    plt.show()

#%% Rsultados 
res_csC=glob(os.path.join('../250716_NE@citrato_250609_c_paper_descongelamiento','cong_sin_campo', '**', '*resultados.txt'),recursive=True)
res_csC.sort()
meta_csC_0, _,_,T_csC_0,_,_,_,_,_,_,_,_,SAR_csC_0,tau_csC_0,_=lector_resultados(res_csC[0])
meta_csC_1, _,_,T_csC_1,_,_,_,_,_,_,_,_,SAR_csC_1,tau_csC_1,_=lector_resultados(res_csC[1])
meta_csC_2, _,_,T_csC_2,_,_,_,_,_,_,_,_,SAR_csC_2,tau_csC_2,_=lector_resultados(res_csC[2])


res_cCT=glob(os.path.join('../250716_NE@citrato_250609_c_paper_descongelamiento','cong_con_campo', '**', '*resultados.txt'),recursive=True)
res_cCT.sort()
meta_cCT_0, _,_,T_cCT_0,_,_,_,_,_,_,_,_,SAR_cCT_0,tau_cCT_0,_=lector_resultados(res_cCT[0])
meta_cCT_1, _,_,T_cCT_1,_,_,_,_,_,_,_,_,SAR_cCT_1,tau_cCT_1,_=lector_resultados(res_cCT[1])
meta_cCT_2, _,_,T_cCT_2,_,_,_,_,_,_,_,_,SAR_cCT_2,tau_cCT_2,_=lector_resultados(res_cCT[2])

res_cCA=glob(os.path.join('38', '**', '*resultados.txt'),recursive=True)
res_cCA.sort()
meta_cCA_0, _,_,T_cCA_0,_,_,_,_,_,_,_,_,SAR_cCA_0,tau_cCA_0,_=lector_resultados(res_cCA[0])
meta_cCA_1, _,_,T_cCA_1,_,_,_,_,_,_,_,_,SAR_cCA_1,tau_cCA_1,_=lector_resultados(res_cCA[1])
meta_cCA_2, _,_,T_cCA_2,_,_,_,_,_,_,_,_,SAR_cCA_2,tau_cCA_2,_=lector_resultados(res_cCA[2])

# %%
fig, (a,b,c)=plt.subplots(nrows=3,constrained_layout=True, sharex=True, sharey=True,figsize=(9,7))

a.set_title('cong s/ campo',loc='left')
a.plot(T_csC_0,tau_csC_0,'.-')
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
#%% Promedio los tau por temperatura
def promediar_tau(T_list, tau_list, nombre_conjunto, intervalo_temp=1):
    """
    Parámetros:
    - T_list: Lista de arrays de temperatura (ej: [T_csC_0, T_csC_1, T_csC_2])
    - tau_list: Lista de arrays de tau correspondientes (ej: [tau_csC_0, tau_csC_1, tau_csC_2])
    - nombre_conjunto: Nombre del conjunto (ej: "csC", "cCT" o "cCA")
    - intervalo_temp: Ancho del intervalo de temperatura (default: 1°C)
    
    Retorna:
    - T_intervalos: Array con los límites inferiores de los intervalos de temperatura
    - prom_tau: Array con los promedios de tau (sin NaN)
    - err_tau: Array con los errores estándar (sin NaN)
    - err_temperatura: Array con el error de temperatura (intervalo_temp/2)
    """
    # Recortar datos entre -20 y 20 °C y concatenar
    T_total = np.array([])
    tau_total = np.array([])
    
    for T, tau in zip(T_list, tau_list):
        idx = np.nonzero((T >= -20) & (T <= 20))
        T_total = np.concatenate((T_total, T[idx]))
        tau_total = np.concatenate((tau_total, tau[idx]))
    
    # Crear intervalos de temperatura
    T_intervalos = np.arange(np.min(T_total), np.max(T_total) + intervalo_temp, intervalo_temp)
    
    # Calcular promedios y errores (ignorando NaN)
    prom_tau = []
    err_tau = []
    
    for temp in T_intervalos:
        mask = (T_total >= temp) & (T_total < temp + intervalo_temp)
        tau_intervalo = tau_total[mask]
        
        # Calcular media y std ignorando NaN
        mean_val = np.nanmean(tau_intervalo)
        std_val = np.nanstd(tau_intervalo)
        
        prom_tau.append(mean_val)
        err_tau.append(std_val)
    
    # Convertir a arrays de numpy y eliminar intervalos con NaN
    prom_tau = np.array(prom_tau)
    err_tau = np.array(err_tau)
    
    # Máscara para valores no NaN
    mask_no_nan = ~np.isnan(prom_tau)
    
    # Filtrar resultados
    T_intervalos = T_intervalos[mask_no_nan]
    prom_tau = prom_tau[mask_no_nan]
    err_tau = err_tau[mask_no_nan]
    
    # Error de temperatura (mitad del intervalo)
    err_temperatura = np.full(len(T_intervalos), intervalo_temp/2)
    
    # Mostrar resultados
    print(f"\nResultados para {nombre_conjunto}:")
    print("Intervalo de Temperatura   |   Promedio de Tau   |   Error Std")
    print("--------------------------------------------------------------")
    for i, temp in enumerate(T_intervalos):
        print(f"{temp:.2f} - {temp + intervalo_temp:.2f} °C   |   {prom_tau[i]:.2e}   |   {err_tau[i]:.2e}")
    
    return T_intervalos, prom_tau, err_tau, err_temperatura

#%% Ejemplo de uso
T_csC, prom_tau_csC, err_tau_csC, err_temp_csC = promediar_tau([T_csC_0, T_csC_1, T_csC_2], [tau_csC_0, tau_csC_1, tau_csC_2], "csC")
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
# %%

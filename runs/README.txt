# runs/

Questa cartella si popola durante l'esecuzione: ogni stage della pipeline scrive qui i propri output,
sotto <pilastro>/<stage>/, insieme ai due snapshot di configurazione config.yaml e
config_effective.yaml.

Il contenuto non è pubblicato: sono molti gigabyte di artefatti. Quelli degli stage dei tre
orchestratori preprocessing.py, dae.py, secondo_stadio.py si rigenerano eseguendo gli stage;
quelli dei driver di tools/ che girarono su cluster (pesi dei finalisti del DAE, compressione,
hardening, ricerca ) richiedono i nodi e i dati esterni descritti in tools/README.txt. I
derivati consegnati stanno in models/ e inference/modelli/.

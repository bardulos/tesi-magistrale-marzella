#!/bin/bash
# Escludi sia file_unito.csv sia file_filtrato.csv dal glob: al secondo run entrambi esistono gia'
# e includerli duplicherebbe i dati nell'output.
ls -1 *.csv | grep -v -e file_unito -e file_filtrato | head -n 1 | xargs head -n 1 > file_unito.csv
ls -1 *.csv | grep -v -e file_unito -e file_filtrato | xargs tail -q -n +2 | tee -a file_unito.csv | wc -l

head -n 1 file_unito.csv > file_filtrato.csv
tail -n +2 file_unito.csv | awk -F',' '$1 != "<IP_LAN_2>" && $1 != "<IP_LAN_1>" && $1 != "<IP_LAN_4>" && $1 != "<IP_LAN_3>" && $2 != "<IP_LAN_3>" && $2 != "<IP_LAN_2>" && $2 != "<IP_LAN_1>" && $2 != "<IP_LAN_4>"' | tee -a file_filtrato.csv | wc -l

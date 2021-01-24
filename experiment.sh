#!/bin/sh
mkdir -p ./log
rm -r ./save/$MD/*
c_MD=$((0))
for MD in "dielectric" "neohookean"
do
    c_POW=$((0))
    for POW in "[0.8, 0.9]" "[0.85, 0.95]"
    do
        echo ==================================================
        echo Experiment  : $MD - $c_POW
        echo Power Init  : $POW
        echo -------------------------------------------------- 
        echo Update Setting...
        sed -i "7s/.*/model = '$MD'/" constants.py
        sed -i "113s/.*/channels = $CH/" constants.py
        echo Running Code...
        # Python print will be logged in log/*.*** file.
        # Run this bash file with './experiment.sh > LOG &' command.
        python3 -u train.py > log/$c_MD.$c_CH$c_LR$c_BZ &
        wait
        echo Done.
        echo ==================================================
        echo Save...
        mkdir -p ./save/$MD/$c_CH$c_LR$c_BZ
        if [ $c_MD -eq 0 ]
        then
            cp -r ./ckpt/$MD/Sylgard186 ./save/$MD/$c_POW/ckpt
        else
            cp -r ./ckpt/$MD ./save/$MD/$c_POW/ckpt
        fi
        cp -r ./plot/$MD ./save/$MD/$c_POW/plot
        echo Done.
        rm -r ./plot/$MD/test/*
        rm -r ./ckpt/$MD/Sylgard186/*
        c_POW=$((c_POW + 1))
    done
    c_MD=$((c_MD + 1))
done


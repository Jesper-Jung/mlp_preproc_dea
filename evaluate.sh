#!/bin/sh
EPOCH=$((200000))
python3 test.py --checkpoint ckpt/dielectric/Sylgard186/$EPOCH/filter.pt --compensate=True
python3 filter.py --checkpoint ckpt/dielectric/Sylgard186/$EPOCH/filter.pt --song=True --sine=True --chirp=True --resample=True

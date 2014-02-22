#!/bin/bash
for f in `ls _static/orig/`; do convert _static/orig/$f -resize 800x500 _static/$f; done
make slides
firefox _build/slides/index.html &

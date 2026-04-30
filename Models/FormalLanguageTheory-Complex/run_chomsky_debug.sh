f=$1

./main \
    --input=data/Chomsky \
    --long-output \
    --alphabet="dNnprITit" \
    --time=10s \
    --nfactors=$1 \
    --top=500 \
    --thin=0 \
    --restart=100000 \
    --maxlength=128 \
    --prN=100 \
    --threads=15 \
    --chains=15 \
    --output="output_f${f}"
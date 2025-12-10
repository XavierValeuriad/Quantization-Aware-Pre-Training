rsync -avz --progress \
    --include='*/' \
    --include='*.tfevents*' \
    --include='evaluation_results.json' \
    --include='*log_history.csv' \
    --include='emissions.csv' \
    --exclude='cache/' \
    --exclude='datasets/' \
    --exclude='eval_temp/' \
    --exclude='sources/' \
    --exclude='*' \
    <jeanzay_id>@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mwd/<jeanzay_id>/ \
    ./../../SCRATCH/
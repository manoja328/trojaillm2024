python metafile_generator.py

SIMG="llm-pretrain-apr2024_test_.simg"
export APPTAINER_TMPDIR=/workspace/manoj/tmp/
sudo -E apptainer build $SIMG detector.def

DIR=`pwd`

singularity run \
--bind $DIR \
--nv ./$SIMG infer \
--model_filepath=/workspace/manoj/trojai-llm2024_rev1/id-00000000 \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=/workspace/manoj/trojai-llm2024_rev1/id-00000000/clean-example-data \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters


~/gdrive files upload $SIMG
echo "=== === === === ==="
echo "upload to google drive ...."
echo "rename as:  llm-pretrain-apr2024_sts_<cotainer_name>.simg"
echo "=== don't forget to share it with trojai@gmail ==="
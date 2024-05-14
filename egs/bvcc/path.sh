# path related
export PRJ_ROOT="${PWD}/../.."
if [ -e "${PRJ_ROOT}/tools/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tools/venv/bin/activate"
fi

MAIN_ROOT=$PWD/../..
export PATH=$MAIN_ROOT/sheet/bin:$PATH

# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg

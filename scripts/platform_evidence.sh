#!/usr/bin/env bash
# Run on the LinuxONE VM and screenshot the output for the video / judges.
cd "$(dirname "$0")/.."
echo "== Architecture =="; uname -a; echo; lscpu | head -20; echo
echo "== Memory =="; free -h; echo
echo "== Python stack =="
python3 -c "import platform,sklearn,pandas,numpy;print(platform.machine(),'python',platform.python_version(),'sklearn',sklearn.__version__,'pandas',pandas.__version__,'numpy',numpy.__version__)"
echo; echo "== Container =="; (podman --version || docker --version) 2>/dev/null
echo; echo "== Model metrics =="
python3 -c "import json;m=json.load(open('reports/metrics.json'));print('trained on',m['platform']);[print('h',h,'persistence MAE',v['persistence']['MAE'],'-> model MAE',v.get('hybrid',v.get('hgb'))['MAE'],'| p90 coverage',v['p90_coverage'],'| train s',v['train_seconds']) for h,v in m['horizons'].items()]"
echo; echo "== IBM Z AI accelerator (NNPA) =="
grep -o -w nnpa /proc/cpuinfo | head -1 || echo "nnpa flag not found"
python3 -c "import torch, torch_nnpa; x=torch.ones(4,4).to('nnpa'); print('torch_nnpa OK, test tensor on', x.device)" 2>&1 | tail -1
[ -f reports/nnpa_benchmark.json ] && python3 -c "import json;r=json.load(open('reports/nnpa_benchmark.json'));[print('h',x['horizon_h'],'CPU',x['cpu'],'| NNPA',x['nnpa']) for x in r['runs']]"
echo; echo "== GenAI (local LLM) =="
python3 -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)" 2>&1 | tail -1
ls -la ${LOCAL_GGUF:-/home/ibm-user/*.gguf} 2>/dev/null

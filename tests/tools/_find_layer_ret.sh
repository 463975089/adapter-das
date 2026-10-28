S=$HOME/sgl_community/sglang
L=$(grep -n "class DeepseekV2DecoderLayer" $S/srt/models/deepseek_v2.py | cut -d: -f1)
# print the forward method of the decoder layer (search forward then 90 lines)
F=$(sed -n "${L},$((L+280))p" $S/srt/models/deepseek_v2.py | grep -n "def forward" | head -1 | cut -d: -f1)
sed -n "$((L+F-1)),$((L+F+95))p" $S/srt/models/deepseek_v2.py

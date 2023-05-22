import pandas as pd

def fmri2words(text_data, Trs, section, delay = 5, window = 0.2) :
  chunks = []
  text = text_data[text_data['section'] == section]
  for tr in range(Trs) :
    onset = tr*2-delay
    offset = onset + 2
    chunk_data = text[(text['onset']>= onset - window) & (text['offset']< offset + window)]
    chunks.append(" ".join(list(chunk_data['word'])))
  return chunks

def text2fmri(textgrid, sent_n, delay = 5) :
  scan_idx = []
  chunks = []
  textgrid = textgrid.tiers
  chunk = ""
  sent_i = 0
  idx_start = int(delay/2)
  for interval in textgrid[0].intervals[1:]:
    if interval.mark == "#":
      chunk+= "."
      if sent_i == sent_n :
        chunks.append(chunk[1:])
        idx_end = min(int((interval.maxTime + delay)/2)+1, 282)
        scan_idx.append(slice(idx_start, idx_end))
        sent_i = 0
        chunk = ""
        idx_start = idx_end - 1
      sent_i += 1
      continue
    chunk += " " + interval.mark
  return chunks, scan_idx


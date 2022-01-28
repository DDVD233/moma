import moma_preproc


def proc_anns(dir_moma, fname_ann_phase1, fname_ann_phase2):
  taxonomy_parser = moma_preproc.TaxonomyParser(dir_moma)
  taxonomy_parser.dump(verbose=False)

  ann_phase1 = moma_preproc.AnnPhase1(dir_moma, fname_ann_phase1)
  ann_phase1.inspect(verbose=False)

  ann_phase2 = moma_preproc.AnnPhase2(dir_moma, fname_ann_phase2)
  ann_phase2.inspect(verbose=False)

  ann_merger = moma_preproc.AnnMerger(dir_moma, ann_phase1, ann_phase2)
  ann_merger.dump()


def proc_videos(dir_moma):
  video_processor = moma_preproc.VideoProcessor(dir_moma)
  video_processor.select()
  video_processor.trim_act()
  video_processor.trim_sact()
  video_processor.trim_hoi()


def main():
  dir_moma = '/home/alan/ssd/moma'
  fname_ann_phase1 = 'video_anns_phase1_processed.json'
  fname_ann_phase2 = 'MOMA-videos-0121-all.jsonl'

  proc_anns(dir_moma, fname_ann_phase1, fname_ann_phase2)
  proc_videos(dir_moma)


if __name__ == '__main__':
  main()
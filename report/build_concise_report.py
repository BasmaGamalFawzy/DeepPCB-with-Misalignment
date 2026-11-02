from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT=Path(__file__).parent
OUT=ROOT/'pcb_data_preparation_report_revised.pdf'
FIG=ROOT/'fig'
NAVY=colors.HexColor('#17324D'); TEAL=colors.HexColor('#147D92'); PALE=colors.HexColor('#EDF4F7'); LINE=colors.HexColor('#CAD5DE'); INK=colors.HexColor('#243746')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',parent=styles['Title'],fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=NAVY,alignment=TA_CENTER,spaceAfter=7))
styles.add(ParagraphStyle(name='SubTitle',parent=styles['Normal'],fontSize=10,leading=14,textColor=TEAL,alignment=TA_CENTER,spaceAfter=11))
styles.add(ParagraphStyle(name='H1x',parent=styles['Heading1'],fontName='Helvetica-Bold',fontSize=15,leading=18,textColor=NAVY,spaceBefore=1,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='H2x',parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=10.5,leading=13,textColor=TEAL,spaceBefore=7,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle(name='Bodyx',parent=styles['BodyText'],fontName='Helvetica',fontSize=9,leading=12,textColor=INK,spaceAfter=5))
styles.add(ParagraphStyle(name='Smallx',parent=styles['BodyText'],fontName='Helvetica',fontSize=7.7,leading=10,textColor=INK,spaceAfter=3))
styles.add(ParagraphStyle(name='Cellx',parent=styles['BodyText'],fontName='Helvetica',fontSize=7.7,leading=9.5,textColor=INK))
styles.add(ParagraphStyle(name='CellHeadx',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=7.7,leading=9.5,textColor=colors.white))
styles.add(ParagraphStyle(name='Calloutx',parent=styles['BodyText'],fontName='Helvetica-Bold',fontSize=8.7,leading=12,textColor=NAVY,backColor=PALE,borderColor=LINE,borderWidth=.5,borderPadding=7,spaceBefore=4,spaceAfter=7))

def P(s,style='Bodyx'): return Paragraph(s,styles[style])
def table(rows,widths,header=True):
    data=[]
    for ri,row in enumerate(rows):
        data.append([P(str(c),'CellHeadx' if header and ri==0 else 'Cellx') for c in row])
    t=Table(data,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    cmds=[('GRID',(0,0),(-1,-1),.35,LINE),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
    if header: cmds += [('BACKGROUND',(0,0),(-1,0),TEAL),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE])]
    else: cmds += [('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,PALE])]
    t.setStyle(TableStyle(cmds)); return t

def footer(c,doc):
    c.saveState(); w,h=A4
    c.setStrokeColor(LINE); c.line(17*mm,13*mm,w-17*mm,13*mm)
    c.setFillColor(colors.HexColor('#627D98')); c.setFont('Helvetica',7.5)
    c.drawString(17*mm,8*mm,'COEN 6314  |  DeepPCB data preparation')
    c.drawRightString(w-17*mm,8*mm,f'{doc.page}')
    c.restoreState()

def image(name,width=175*mm):
    im=Image(str(FIG/name)); im._restrictSize(width,155*mm); im.hAlign='CENTER'; return im

story=[]
# Page 1
story += [Spacer(1,8*mm),P('PCB Defect Detection Under Image Misalignment','ReportTitle'),P('Data Preparation Report  |  COEN 6314 Course Project  |  October 2026','SubTitle'),P('Basma Kurany  |  Concordia University, Gina Cody School of Engineering and Computer Science','SubTitle'),Spacer(1,3*mm)]
story += [P('Summary','H1x'),P('This report describes the DeepPCB data and the reproducible preparation pipeline for measuring the effect of image misalignment on defect detection. It covers the source audit, group-aware split, controlled shifts, label handling, checks, and model-ready exports. Preparation results are reported here; detector performance is not. Proposal V4 lists Fast R-CNN, YOLOv10-Nano, and SSD-MobileNet as detector options. This project selects YOLOv10-Nano for the implementation.','Bodyx')]
story += [P('Dataset at a glance','H2x'),table([['Measure','Value'],['Image pairs','1,500: official 1,000 trainval and 500 test'],['Image size','640 × 640 pixels'],['Annotations','10,013 bounding boxes across six defect classes'],['Classes','Open, short, mousebite, spur, copper, pin-hole'],['Prepared image format','Grayscale PNG; reference stays fixed'],['Groups','11 source groups; 332 test pairs from four groups absent in trainval']], [48*mm,126*mm])]
story += [P('Source and labels','H2x'),P('DeepPCB contains template/inspection pairs cropped from larger board scans and initially aligned by template matching. Each pair has a reference image, an inspected image, and a text label file with box corners and a class ID. Most annotated defects were added artificially by the dataset authors, so results may not transfer directly to real production defects. The extracted source tree is under <font name="Courier">Data/DeepPCB-master/DeepPCB-master/PCBData</font>.','Bodyx')]
story += [table([['Partition','Pairs','Open','Short','Mousebite','Spur','Copper','Pin-hole'],['Trainval','1,000','1,283','1,028','1,379','1,142','1,010','1,031'],['Test','500','659','478','586','483','464','470'],['Total','1,500','1,942','1,506','1,965','1,625','1,474','1,501']], [24*mm,21*mm,20*mm,20*mm,26*mm,19*mm,22*mm,22*mm])]
story += [P('Boxes range from 18–173 px wide (median 35) and 19–152 px high (median 33). Fifty-seven pairs contain at least one 3-channel JPEG; prepared outputs are converted to grayscale PNG.','Smallx'),PageBreak()]
# Page 2
story += [P('Preparation pipeline','H1x'),P('The original data remain unchanged. The pipeline indexes and audits the source, creates a grouped validation split, generates seeded evaluation conditions, verifies image/label geometry, and exports COCO and YOLO data. Seed 6314 combined with the pair ID makes transforms independent of file order.','Bodyx')]
story += [table([['Scripts','Purpose'],['01–03: inspect, duplicate scan, alignment audit','Build source indexes; check files, candidate duplicates, and native offsets.'],['04: make splits','Create group-aware train/validation/test manifests and duplicate flags.'],['05 + pcb_shift','Create transformed image sets and boxes; provide training jitter utilities.'],['06: QA checks','Run geometry checks and generate visual check sheets.'],['07–09: COCO and YOLO export/check','Export model-ready annotations and images; verify the YOLO package.'],['run_all.sh','Run the complete preparation pipeline.']], [63*mm,111*mm])]
story += [P('Data audit','H2x'),P('Native alignment: an integer offset is estimated by minimizing template/inspection XOR disagreement over offsets up to ±8 px. A pair is flagged when the best non-zero offset reduces disagreement by at least 1.5×. This heuristic flags 123/1,500 pairs (75 train, 8 validation, 40 test), with offsets up to 8 px. Therefore, <b>s0 means “as supplied,” not guaranteed perfectly aligned.</b>','Bodyx'),image('native_misalignment_examples.png',width=168*mm),P('Native misalignment examples. Red and green indicate pixels unique to each image; yellow indicates overlap.','Smallx')]
# Page 3
story += [P('Duplicates and data partitions','H1x'),P('A difference-hash scan produced 337 candidates. Pixel-level comparison after shifts up to 8 px confirmed 40 duplicate pairs, all within the same source group. Three test pairs (50600065, 92000114, 92000118) duplicate training pairs and are flagged for a separate sensitivity analysis.','Bodyx'),P('Grouped split','H2x'),P('The official 500-pair test set is retained. One whole trainval group, group00041 (200 pairs), is held out for validation; the remaining 800 pairs form training data. This keeps transformed copies together. The validation group was selected by minimizing the L1 difference in class proportions among candidate group-held-out splits containing 17–23% of trainval.','Bodyx')]
story += [table([['Split','Pairs','Native-offset pairs','Group assignment'],['Train','800','75','13000, 20085, 44000, 50600, 77000, 92000'],['Validation','200','8','00041'],['Test','500','40','All 11 official test groups']], [28*mm,24*mm,39*mm,83*mm])]
story += [P('Primary test results should include all 500 official test pairs. A clean-pair sensitivity analysis may exclude the 40 natively misaligned test pairs, leaving 460. The count 1,377 refers to clean pairs across the full dataset, not the test set.','Calloutx'),P('The test set also contains 332 pairs from four groups absent from trainval. This provides an unseen-group stress test, while making the test group distribution different from training.','Bodyx'),PageBreak()]
# Page 4
story += [P('Controlled misalignment and labels','H1x'),P('Validation and test each contain ten conditions: s0 (as supplied); horizontal and vertical shifts of 1, 2, 3, and 5 px; and a residual condition combining a 0.25–1.5 px translation with rotation between −0.5° and +0.5°. Training stores only s0; augmentation is generated separately.','Bodyx'),P('For every condition, the reference stays unchanged and the same seeded transform is applied to the inspected image and its boxes. Integer shifts use exact pixel sampling; residual transforms use bilinear interpolation. For rotation, all four box corners are transformed and enclosed by a tight axis-aligned box.','Bodyx'),P('Border and ignore handling','H2x'),P('Evaluation excludes an 8 px border. Boxes within 14 px of an original image edge are marked ignore in every condition: 258/10,013 boxes (2.6%). Treat these as “don’t care” annotations during scoring.','Bodyx'),Table([[image('check_s5x.png',width=82*mm),image('check_r.png',width=82*mm)]],colWidths=[86*mm,86*mm],style=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]),P('Visual label checks for a 5 px shift (left) and residual transform (right).','Smallx'),PageBreak()]
# Page 5
story += [P('Exports and YOLO input','H1x'),P('CSV manifests are the audit-friendly source of truth for pair IDs, groups, splits, transform parameters, paths, boxes, classes, and ignore flags. COCO JSON exports provide one file per available split/condition (21 files). The original DeepPCB text labels remain unchanged.','Bodyx'),table([['Artifact','Contents'],['sets/ref/','1,500 unchanged reference PNGs.'],['sets/{split}/{condition}/','7,800 grayscale inspection PNGs.'],['index/pairs.csv, boxes.csv, splits.csv','Source inventory, annotations, and partition metadata.'],['index/images.csv, labels.csv','Per-image transform metadata and per-box labels/ignore flags.'],['index/coco/','COCO exports for each available split/condition.'],['yolo_train.zip, yolo_eval_*.zip','YOLO training and evaluation composites with labels.'],['reports/, qa/','Machine-readable summaries and visual checks.']], [67*mm,107*mm])]
story += [P('YOLOv10-Nano data path','H2x'),P('YOLO inputs use three channels: reference, inspected image, and absolute difference. Labels use normalized <font name="Courier">class xc yc w h</font> rows. The training package contains 800 unaugmented images and 2,400 augmented images (original, one random integer shift, and one residual transform per pair). Disable independent HSV channel augmentation because it corrupts the difference channel. YOLO labels cannot encode the ignore flag, so evaluation must apply the ignore rule from the master CSV.','Bodyx'),image('yolo_check.png',width=165*mm),P('Export check for original, shifted, and residual composites from one pair.','Smallx'),PageBreak()]
# Page 6
story += [P('Verification and next steps','H1x'),table([['Automated check','Result'],['Integer-shift pixel checks','5,600 / 5,600 passed'],['Transformed box-content checks','36,640 / 36,640 passed'],['Non-ignored boxes inside valid region','41,220 / 41,220 passed'],['Residual transform audit (120 samples)','Median corner error 0.002 px; maximum 0.05 px'],['YOLO label presence, counts, and classes','10,200 / 10,200 passed'],['Composite channel construction (342 samples)','342 / 342 passed']], [88*mm,86*mm])]
story += [P('The YOLO round-trip check reports coordinate differences for ignored boxes clipped at image boundaries. Evaluation should use the master annotations and apply the documented ignore rule. Rebuild the preparation outputs with Python 3, NumPy, OpenCV, and <font name="Courier">bash MyData/scripts/run_all.sh</font>; set <font name="Courier">PCB_ROOT</font> if the raw-data folder is elsewhere.','Bodyx'),P('Limitations','H2x'),P('Most defects are synthetic; 123 pairs have native alignment offsets; validation consists of one source group; and each validation/test pair has only one residual variant. Three train/test duplicate pairs also warrant a separately labeled sensitivity analysis.','Bodyx'),P('Next steps','H2x'),P('Implement registered and unregistered differencing baselines, train aligned-only and jitter-trained YOLO models, tune thresholds on validation at a common target recall, then evaluate all 500 official test pairs. Report the 460-pair clean-test subset and duplicate-excluded results only as sensitivity analyses. YOLOv10-Nano is the selected detector from the options listed in Proposal V4.','Bodyx'),P('Reference','H2x'),P('[1] S. Tang, F. He, X.-L. Huang, and J. Yang, “Online PCB defect detector on a new PCB defect dataset,” arXiv:1902.06197, 2019.','Smallx')]

doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=19*mm,title='PCB Defect Detection Under Image Misalignment - Data Preparation Report',author='Basma Kurany')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT.resolve())






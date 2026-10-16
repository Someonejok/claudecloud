"""Revised manuscript and response-to-reviewers letter (.docx) from results_summary.json.

Formatting follows the journal's requirements and reviewer major 1: Times New Roman
12 pt, bold headings, single spacing, and no section numbers, bullet points or numbered
lists anywhere. Every number is read from results_summary.json; a value that has not
been computed yet is printed as a yellow-highlighted [pending: ...] marker, so nothing
in the text can be a number that was not measured. Sentences that interpret a result
(direction, significance) are generated from the statistics, not written in advance.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

PEND = "\x00"

# ----------------------------------------------------------------------------- references
REFS = {
    "maas2022": "Maas AIR, Menon DK, Manley GT, et al. Traumatic brain injury: progress and challenges in prevention, clinical care, and research. Lancet Neurol. 2022;21(11):1004-1060. doi:10.1016/S1474-4422(22)00309-X",
    "dewan2019": "Dewan MC, Rattani A, Gupta S, et al. Estimating the global incidence of traumatic brain injury. J Neurosurg. 2019;130(4):1080-1097. doi:10.3171/2017.10.JNS17352",
    "gbd2019": "GBD 2016 Traumatic Brain Injury and Spinal Cord Injury Collaborators. Global, regional, and national burden of traumatic brain injury and spinal cord injury, 1990-2016: a systematic analysis for the Global Burden of Disease Study 2016. Lancet Neurol. 2019;18(1):56-87. doi:10.1016/S1474-4422(18)30415-0",
    "hssayeni2020": "Hssayeni MD, Croock MS, Salman AD, Al-khafaji HF, Yahya ZA, Ghoraani B. Intracranial hemorrhage segmentation using a deep convolutional model. Data. 2020;5(1):14. doi:10.3390/data5010014",
    "physionet": "Hssayeni M. Computed tomography images for intracranial hemorrhage detection and segmentation (version 1.3.1). PhysioNet. 2020. doi:10.13026/4nae-zg36",
    "goldberger2000": "Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation. 2000;101(23):e215-e220. doi:10.1161/01.CIR.101.23.e215",
    "wu2023bhsd": "Wu B, Xie Y, Zhang Z, Ge J, Yaxley K, Bahadir S, Wu Q, Liu Y, To M-S. BHSD: a 3D multi-class brain hemorrhage segmentation dataset. In: Machine Learning in Medical Imaging (MLMI 2023). Lecture Notes in Computer Science, vol 14348. Springer; 2023:147-156. doi:10.1007/978-3-031-45673-2_15",
    "li2023instance": "Li X, Luo G, Wang K, et al. The state-of-the-art 3D anisotropic intracranial hemorrhage segmentation on non-contrast head CT: the INSTANCE challenge. arXiv:2301.03281 [preprint]. 2023",
    "chilamkurthy2018": "Chilamkurthy S, Ghosh R, Tanamala S, et al. Deep learning algorithms for detection of critical findings in head CT scans: a retrospective study. Lancet. 2018;392(10162):2388-2396. doi:10.1016/S0140-6736(18)31645-3",
    "kuo2019": "Kuo W, Häne C, Mukherjee P, Malik J, Yuh EL. Expert-level detection of acute intracranial hemorrhage on head computed tomography using deep learning. Proc Natl Acad Sci U S A. 2019;116(45):22737-22745. doi:10.1073/pnas.1908021116",
    "flanders2020": "Flanders AE, Prevedello LM, Shih G, et al. Construction of a machine learning dataset through collaboration: the RSNA 2019 brain CT hemorrhage challenge. Radiol Artif Intell. 2020;2(3):e190211. doi:10.1148/ryai.2020190211",
    "wang2021": "Wang X, Shen T, et al. A deep learning algorithm for automatic detection and classification of acute intracranial hemorrhages in head CT scans. NeuroImage Clin. 2021;32:102785. doi:10.1016/j.nicl.2021.102785",
    "arbabshirani2018": "Arbabshirani MR, Fornwalt BK, Mongelluzzo GJ, et al. Advanced machine learning in action: identification of intracranial hemorrhage on computed tomography scans of the head with clinical workflow integration. NPJ Digit Med. 2018;1:9. doi:10.1038/s41746-017-0015-z",
    "oneill2021": "O'Neill TJ, et al. Active reprioritization of the reading worklist using artificial intelligence has a beneficial effect on the turnaround time for interpretation of head CT with intracranial hemorrhage. Radiol Artif Intell. 2021;3(2):e200024. doi:10.1148/ryai.2020200024",
    "ginat2020": "Ginat DT. Analysis of head CT scans flagged by deep learning software for acute intracranial hemorrhage. Neuroradiology. 2020;62(3):335-340. doi:10.1007/s00234-019-02330-w",
    "monteiro2020": "Monteiro M, Newcombe VFJ, Mathieu F, et al. Multiclass semantic segmentation and quantification of traumatic brain injury lesions on head CT using deep learning: an algorithm development and multicentre validation study. Lancet Digit Health. 2020;2(6):e314-e322. doi:10.1016/S2589-7500(20)30085-6",
    "inkeaw2022": "Inkeaw P, et al. Automatic hemorrhage segmentation on head CT scan for traumatic brain injury using 3D deep learning model. Comput Biol Med. 2022;146:105530. doi:10.1016/j.compbiomed.2022.105530",
    "spahr2023": "Spahr A, Ståhle J, Wang C, Kaijser M. Label-efficient deep semantic segmentation of intracranial hemorrhages in CT-scans. Front Neuroimaging. 2023;2:1157565. doi:10.3389/fnimg.2023.1157565",
    "rasoulian2023": "Rasoulian A, Salari S, Xiao Y. Weakly supervised intracranial hemorrhage segmentation using head-wise gradient-infused self-attention maps from a Swin transformer in categorical learning. Mach Learn Biomed Imaging. 2023;2:338-360. doi:10.59275/j.melba.2023-553a",
    "zheng2025": "Zheng C, Xie G, Wang H, et al. HemaContour: explicit parametric contour learning for robust ICH segmentation on non-contrast CT. NPJ Digit Med. 2025;8:754. doi:10.1038/s41746-025-02124-w",
    "ronneberger2015": "Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015). Lecture Notes in Computer Science, vol 9351. Springer; 2015:234-241. doi:10.1007/978-3-319-24574-4_28",
    "isensee2021": "Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods. 2021;18(2):203-211. doi:10.1038/s41592-020-01008-z",
    "isensee2024": "Isensee F, Wald T, Ulrich C, Baumgartner M, Roy S, Maier-Hein K, Jäger PF. nnU-Net revisited: a call for rigorous validation in 3D medical image segmentation. In: MICCAI 2024. Lecture Notes in Computer Science, vol 15009. Springer; 2024:488-498",
    "he2016": "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016:770-778. doi:10.1109/CVPR.2016.90",
    "srikrishna2022": "Srikrishna M, Heckemann RA, Pereira JB, et al. Comparison of two-dimensional- and three-dimensional-based U-Net architectures for brain tissue classification in one-dimensional brain CT. Front Comput Neurosci. 2022;15:785244. doi:10.3389/fncom.2021.785244",
    "maierhein2024": "Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. Nat Methods. 2024;21(2):195-212. doi:10.1038/s41592-023-02151-z",
    "reinke2024": "Reinke A, Tizabi MD, et al. Understanding metric-related pitfalls in image analysis validation. Nat Methods. 2024;21(2):182-194. doi:10.1038/s41592-023-02150-0",
    "taha2015": "Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging. 2015;15:29. doi:10.1186/s12880-015-0068-x",
    "varoquaux2022": "Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. NPJ Digit Med. 2022;5:48. doi:10.1038/s41746-022-00592-y",
    "kapoor2023": "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804. doi:10.1016/j.patter.2023.100804",
    "eelbode2020": "Eelbode T, Bertels J, Berman M, et al. Optimization for medical image segmentation: theory and practice when evaluating with Dice score or Jaccard index. IEEE Trans Med Imaging. 2020;39(11):3679-3690. doi:10.1109/TMI.2020.3002417",
    "yeung2022": "Yeung M, Sala E, Schönlieb C-B, Rundo L. Unified focal loss: generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation. Comput Med Imaging Graph. 2022;95:102026. doi:10.1016/j.compmedimag.2021.102026",
    "salehi2017": "Salehi SSM, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. In: Machine Learning in Medical Imaging (MLMI 2017). Lecture Notes in Computer Science, vol 10541. Springer; 2017:379-387",
    "chlap2021": "Chlap P, Min H, Vandenberg N, Dowling J, Holloway L, Haworth A. A review of medical image data augmentation techniques for deep learning applications. J Med Imaging Radiat Oncol. 2021;65(5):545-563. doi:10.1111/1754-9485.13261",
    "shorten2019": "Shorten C, Khoshgoftaar TM. A survey on image data augmentation for deep learning. J Big Data. 2019;6:60. doi:10.1186/s40537-019-0197-0",
    "wilcoxon1945": "Wilcoxon F. Individual comparisons by ranking methods. Biometrics Bull. 1945;1(6):80-83. doi:10.2307/3001968",
    "holm1979": "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70",
    "page1963": "Page EB. Ordered hypotheses for multiple treatments: a significance test for linear ranks. J Am Stat Assoc. 1963;58(301):216-230. doi:10.1080/01621459.1963.10500843",
    "cliff1993": "Cliff N. Dominance statistics: ordinal analyses to answer ordinal questions. Psychol Bull. 1993;114(3):494-509. doi:10.1037/0033-2909.114.3.494",
    "denotter": "DenOtter TD, Schubert J. Hounsfield unit. In: StatPearls. StatPearls Publishing; 2024. NCBI Bookshelf NBK547721",
    "tfbinaryiou": "TensorFlow. tf.keras.metrics.BinaryIoU. API documentation. https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryIoU",
    "smp": "Iakubovskii P. Segmentation Models Pytorch. GitHub repository. https://github.com/qubvel-org/segmentation_models.pytorch",
}

TABLES = ["partitions", "subtypes", "balance", "architecture", "test", "nnunet_cv", "imbalance", "window",
          "augmentation", "trend", "morphology", "compute", "subtype_test", "size_test", "failure"]
FIGURES = ["overlays", "learning", "dose", "qualitative"]


# ----------------------------------------------------------------------------- values
class Values:
    def __init__(self, S: dict):
        self.S = S
        self.missing = []

    def get(self, path):
        cur = self.S
        for tok in re.split(r"\.(?![^\[]*\])", path):
            m = re.fullmatch(r"([^\[]+)(?:\[(\w+)=([^\]]+)\])?", tok)
            if not m or cur is None:
                return None
            key, sk, sv = m.groups()
            cur = cur.get(key) if isinstance(cur, dict) else None
            if sk is not None:
                cur = next((r for r in (cur or []) if str(r.get(sk)) == sv), None)
        return cur

    def __call__(self, path, fmt="f3"):
        x = self.get(path)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            self.missing.append(path)
            return f"{PEND}[pending: {path.split('.')[-1]}]{PEND}"
        return fmt_value(x, fmt)

    def has(self, path):
        x = self.get(path)
        return x is not None and not (isinstance(x, float) and math.isnan(x))


def fmt_value(x, fmt):
    if fmt == "ci":
        if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            return f"{x[0]:.3f}-{x[1]:.3f}"
        return str(x)
    if fmt == "int":
        return f"{int(round(x)):,}"
    if fmt == "pct":
        return f"{100 * x:.1f} %"
    if fmt == "p":
        return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"
    if fmt.startswith("f"):
        return f"{float(x):.{int(fmt[1:])}f}"
    return str(x)


def slug(name):
    return name.replace(" ", "_").replace("+", "plus").replace("-", "")


HEAD = slug("nnU-Net P+B")
PONLY = slug("nnU-Net P")


# ----------------------------------------------------------------------------- docx layer
class Writer:
    def __init__(self, S, figures_dir: Path):
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt, RGBColor
        self.Inches, self.Pt, self.RGB, self.ALIGN = Inches, Pt, RGBColor, WD_ALIGN_PARAGRAPH
        self.doc = Document()
        self.v = Values(S)
        self.S = S
        self.figdir = Path(figures_dir)
        self.cited = []
        self._style()

    def _style(self):
        from docx.oxml.ns import qn
        for sec in self.doc.sections:
            for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
                setattr(sec, side, self.Inches(1))
        for name in ("Normal", "Title", "Heading 1", "Heading 2", "Caption", "Table Grid"):
            try:
                st = self.doc.styles[name]
            except KeyError:
                continue
            f = st.font
            f.name = "Times New Roman"
            f.color.rgb = self.RGB(0, 0, 0)
            if name != "Table Grid":
                f.size = self.Pt(12)
            rpr = st.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = rpr.makeelement(qn("w:rFonts"), {})
                rpr.append(rf)
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(att), None)
            for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(att), "Times New Roman")
            if name.startswith("Heading") or name == "Title":
                f.bold = True
                st.paragraph_format.space_before = self.Pt(12)
                st.paragraph_format.space_after = self.Pt(6)
            st.paragraph_format.line_spacing = 1.0
        self.doc.styles["Normal"].paragraph_format.space_after = self.Pt(6)

    # citations: [@key] or [@a;@b]
    def _cite(self, text):
        def rep(m):
            keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
            nums = []
            for k in keys:
                if k not in REFS:
                    raise KeyError(f"unknown reference {k}")
                if k not in self.cited:
                    self.cited.append(k)
                nums.append(self.cited.index(k) + 1)
            return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"
        return re.sub(r"\[(@[^\]]+)\]", rep, text)

    def _refs(self, text):
        text = re.sub(r"\{tabn:(\w+)\}", lambda m: f"{TABLES.index(m.group(1)) + 1}", text)
        text = re.sub(r"\{tab:(\w+)\}", lambda m: f"Table {TABLES.index(m.group(1)) + 1}", text)
        return re.sub(r"\{fig:(\w+)\}", lambda m: f"Figure {FIGURES.index(m.group(1)) + 1}", text)

    def p(self, text, bold=False, italic=False, align=None, size=None):
        from docx.enum.text import WD_COLOR_INDEX
        text = self._refs(self._cite(text))
        para = self.doc.add_paragraph()
        if align == "center":
            para.alignment = self.ALIGN.CENTER
        for i, seg in enumerate(text.split(PEND)):
            if not seg:
                continue
            r = para.add_run(seg)
            r.bold, r.italic = bold, italic
            if size:
                r.font.size = self.Pt(size)
            if i % 2 == 1:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return para

    def h1(self, text):
        self.doc.add_paragraph(text, style="Heading 1")

    def h2(self, text):
        self.doc.add_paragraph(text, style="Heading 2")

    def table(self, key, caption, header, rows, note=None):
        n = TABLES.index(key) + 1
        self.p(f"Table {n}. {caption}", bold=True)
        if not rows:
            self.p(f"{PEND}[pending: Table {n} is filled when the corresponding experiment has run]{PEND}")
            return
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        for j, h in enumerate(header):
            c = t.rows[0].cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(str(h))
            run.bold = True
            run.font.size = self.Pt(9)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                c = t.rows[i].cells[j]
                c.text = ""
                run = c.paragraphs[0].add_run(_cell(val))
                run.font.size = self.Pt(9)
        if note:
            self.p(note, italic=True, size=10)

    def table_continued(self, header, rows):
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        for j, h in enumerate(header):
            run = t.rows[0].cells[j].paragraphs[0].add_run(str(h))
            run.bold = True
            run.font.size = self.Pt(9)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                t.rows[i].cells[j].paragraphs[0].add_run(_cell(val)).font.size = self.Pt(9)

    def figure(self, key, filename, caption):
        n = FIGURES.index(key) + 1
        f = self.figdir / filename
        if f.exists():
            self.doc.add_picture(str(f), width=self.Inches(6.3))
            self.doc.paragraphs[-1].alignment = self.ALIGN.CENTER
        else:
            self.p(f"{PEND}[pending: figure file {filename}]{PEND}")
        self.p(f"Figure {n}. {caption}", italic=False, size=11)

    def references(self):
        self.h1("References")
        for i, k in enumerate(self.cited, start=1):
            self.p(f"[{i}] {REFS[k]}")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))


def _cell(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"{v[0]:.3f}-{v[1]:.3f}"
    return str(v)


# ----------------------------------------------------------------------------- generated sentences
def _test_row(S, model, part, tag="recipe"):
    return ((S.get("test") or {}).get(slug(model)) or {}).get(f"{part}_{tag}")


def _cmp_sentence(S, exp, label_of, metric_name="volumetric Dice"):
    """One sentence per comparison vs the reference arm, stating only what the test supports."""
    tests = ((S.get("cv") or {}).get(exp) or {}).get("tests") or []
    out = []
    for t in tests:
        a = label_of(t["A"])
        d, lo, hi = t.get("mean_diff"), t.get("ci95_lo"), t.get("ci95_hi")
        if d is None or (isinstance(d, float) and math.isnan(d)):
            continue
        ph = t.get("p_holm", float("nan"))
        if not t.get("powered"):
            verdict = "the comparison has too few matched blocks to reach significance and is reported descriptively"
        elif t.get("significant"):
            verdict = f"a significant {'improvement' if d > 0 else 'decrease'} after Holm correction ({fmt_value(ph, 'p')})"
        else:
            verdict = f"not significant after Holm correction ({fmt_value(ph, 'p')})"
        out.append(f"{a}: {d:+.3f} {metric_name} (95 % CI {lo:+.3f} to {hi:+.3f}; Cliff's delta "
                   f"{t.get('cliffs_delta', float('nan')):+.2f}; {t.get('n_pairs')} matched blocks), {verdict}")
    return "; ".join(out) + "." if out else ""


def _trend_sentence(S):
    tr = ((S.get("cv") or {}).get("E4_augmentation") or {}).get("trend") or []
    parts = []
    for r in tr:
        p = r.get("p")
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        parts.append(f"{r['family']} ({'decreasing trend, ' + fmt_value(p, 'p') if p < 0.05 else 'no significant monotone decrease, ' + fmt_value(p, 'p')})")
    return ("Page's test for a monotone decrease with magnitude gave: " + "; ".join(parts) + ".") if parts else ""


def _num(S, path):
    v = Values(S).get(path)
    return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def _conclusion_sentence(S):
    iou = _num(S, "test.paper.physionet_recipe.iou_slice_mean")
    head = _num(S, f"test.{HEAD}.physionet_recipe.dice_vol_mean")
    parts = []
    if iou is not None:
        parts.append(f"Under scan-level validation with lesion-only metrics, the original model's mean per-slice "
                     f"lesion IoU on held-out scans was {iou:.3f}, against the 0.85 background-averaged figure "
                     f"reported earlier")
    if head is not None:
        parts.append(f"the best model reached a volumetric Dice of {head:.3f}")
    tr = ((S.get("cv") or {}).get("E4_augmentation") or {}).get("trend") or []
    sig = [r["family"] for r in tr if isinstance(r.get("p"), (int, float)) and r["p"] < 0.05]
    if tr:
        parts.append(("augmentation performance fell significantly with magnitude for " + ", ".join(sig))
                     if sig else "no augmentation family showed a significant monotone decrease with magnitude")
    return ("; ".join(parts) + ".") if parts else f"{PEND}[pending: conclusion depends on the results]{PEND}"


def _data_sentence(S):
    a = _num(S, f"test.{PONLY}.physionet_recipe.dice_vol_mean")
    b = _num(S, f"test.{HEAD}.physionet_recipe.dice_vol_mean")
    ea = _num(S, f"test.{PONLY}.bhsd_recipe.dice_vol_mean")
    eb = _num(S, f"test.{HEAD}.bhsd_recipe.dice_vol_mean")
    if a is None or b is None:
        return f"{PEND}[pending: comparison of nnU-Net trained with and without BHSD]{PEND}"
    s = (f"Adding the BHSD training scans changed volumetric Dice on the PhysioNet test scans from {a:.3f} to "
         f"{b:.3f}")
    if ea is not None and eb is not None:
        s += f", and on the BHSD test scans from {ea:.3f} (external, never seen in training) to {eb:.3f}"
    return s + "; with 15 test scans this comparison is descriptive, and its confidence intervals are in {tab:test}."


# ----------------------------------------------------------------------------- manuscript
def build_manuscript(S: dict, figures_dir: Path, out: Path):
    W = Writer(S, figures_dir)
    v = W.v
    pnt = _test_row(S, "nnU-Net P+B", "physionet")
    d = S.get("data") or {}
    parts = {r["partition"]: r for r in d.get("partitions", [])}
    pn_scans = sum(parts.get(k, {}).get("scans", 0) for k in ("pn_dev", "pn_test")) or None
    bh_scans = sum(parts.get(k, {}).get("scans", 0) for k in ("bh_train", "bh_test")) or None
    S.setdefault("derived", {})
    S["derived"].update({"pn_scans": pn_scans, "bh_scans": bh_scans})

    W.p("Segmentation of Traumatic Intracranial Haemorrhage on Head CT with 2D U-Nets: Augmentation "
        "Intensity, Class Imbalance and Generalisation under Scan-Level Validation", bold=True, align="center", size=14)
    W.p("Arjun Choudhary¹, Tyler Poore²", align="center")
    W.p("¹Inventure Academy, Whitefield, Bengaluru, Karnataka, India. ²Inspirit AI.", align="center")
    W.p("Corresponding author: arjunc.9876@gmail.com", align="center")
    W.p("Keywords: traumatic brain injury, intracranial haemorrhage, computed tomography, deep learning, "
        "U-Net, image segmentation, data augmentation, class imbalance, external validation")

    # ---------------------------------------------------------------- abstract
    W.h1("Abstract")
    W.p("Background. Traumatic intracranial haemorrhage is a time-critical finding on head CT, and its "
        "segmentation supports detection, localisation and volume measurement. Published results on small "
        "public datasets are difficult to compare because evaluation protocols differ, and the effect of "
        "augmentation intensity on segmentation of thin extra-axial haemorrhages has not been measured with "
        "single-factor experiments.")
    W.p(f"Methods. We used {v('derived.pn_scans', 'int')} non-contrast head CT scans from patients with "
        f"traumatic brain injury (PhysioNet CT-ICH) and {v('derived.bh_scans', 'int')} scans from a second public "
        "dataset with voxel-level labels for five haemorrhage subtypes (BHSD). All partitioning was done at the "
        "scan level: 15 PhysioNet scans and the BHSD test scans were held out and evaluated once, after every "
        "setting had been frozen. The headline model was nnU-Net in its 2D configuration trained on PhysioNet "
        "development scans and BHSD training scans. Ablations with a 2D U-Net (five-fold scan-level "
        "cross-validation, paired statistics) varied augmentation intensity one transform family at a time, "
        "the class-imbalance strategy, the CT window and the encoder. The primary metric was per-scan volumetric "
        "Dice on haemorrhage-positive scans; per-slice, lesion-level, boundary and detection metrics are also "
        "reported.")
    W.p(f"Results. On the held-out PhysioNet test scans the headline model reached a volumetric Dice of "
        f"{v(f'test.{HEAD}.physionet_recipe.dice_vol_mean')} (95 % CI "
        f"{v(f'test.{HEAD}.physionet_recipe.dice_vol_mean_ci95', 'ci')}), lesion-wise sensitivity "
        f"{v(f'test.{HEAD}.physionet_recipe.lesion_sensitivity')} with "
        f"{v(f'test.{HEAD}.physionet_recipe.fp_lesions_per_scan', 'f1')} false-positive components per scan, and "
        f"scan-level detection AUC {v(f'test.{HEAD}.physionet_recipe.scan_auc')}; on the BHSD test scans its "
        f"volumetric Dice was {v(f'test.{HEAD}.bhsd_recipe.dice_vol_mean')}. Trained on PhysioNet alone, nnU-Net "
        f"reached {v(f'test.{PONLY}.physionet_recipe.dice_vol_mean')} on the PhysioNet test scans and "
        f"{v(f'test.{PONLY}.bhsd_recipe.dice_vol_mean')} on the external BHSD scans. The original manuscript's "
        f"U-Net and training recipe, evaluated under the same protocol, reached "
        f"{v('test.paper.physionet_recipe.dice_vol_mean')} volumetric Dice and "
        f"{v('test.paper.physionet_recipe.iou_slice_mean')} mean per-slice lesion IoU. "
        + (_trend_sentence(S) or f"{PEND}[pending: augmentation trend result]{PEND}"))
    W.p("Conclusions. " + _conclusion_sentence(S) + " The models are segmentation tools evaluated "
        "retrospectively on public data; they are not validated for clinical decision-making.")

    # ---------------------------------------------------------------- introduction
    W.h1("Introduction")
    W.p("Traumatic brain injury is a leading cause of injury-related death and disability worldwide [@maas2022], "
        "with an estimated 69 million new injuries each year [@dewan2019] and a large burden of prevalent "
        "disability [@gbd2019]. Non-contrast CT is the first-line examination after head trauma, and the "
        "presence, type and size of intracranial haemorrhage determine whether a patient needs neurosurgical "
        "intervention. Traumatic haemorrhage is heterogeneous: epidural and subdural haematomas are extra-axial "
        "collections that lie against the inner table of the skull, subarachnoid and intraventricular "
        "haemorrhage follow the cerebrospinal-fluid spaces, and intraparenchymal haemorrhage and contusions lie "
        "within brain tissue [@monteiro2020;@inkeaw2022]. These lesion types differ in shape, size and clinical "
        "meaning.")
    W.p("This study concerns traumatic intracranial haemorrhage. It is distinct from spontaneous intracerebral "
        "haemorrhage, a subtype of stroke with its own epidemiology; epidemiological figures for intracerebral "
        "haemorrhage do not describe the population studied here, which is drawn from patients with traumatic "
        "brain injury [@hssayeni2020] and includes epidural and subdural haematomas, which are not strokes.")
    W.p("Deep learning detects intracranial haemorrhage on head CT with high accuracy when trained on large "
        "datasets [@chilamkurthy2018;@kuo2019;@wang2021], and the RSNA 2019 challenge released more than 25,000 "
        "annotated examinations for detection and subtype classification [@flanders2020]. In clinical "
        "deployments, reprioritising the reading worklist shortened the time to diagnosis "
        "[@arbabshirani2018;@oneill2021], although accuracy varied across clinical settings [@ginat2020]. "
        "Voxel-level segmentation is harder and depends on far smaller annotated datasets. On the PhysioNet "
        "CT-ICH dataset the original authors reported a Dice coefficient of 0.31 with a 2D U-Net under five-fold "
        "cross-validation [@hssayeni2020]. Transfer learning from unlabelled or weakly labelled scans improves "
        "haemorrhage segmentation, most for small labelled sets [@spahr2023], and segmentation learned from "
        "image-level labels alone can approach fully supervised U-Net performance [@rasoulian2023]. A "
        "contour-based model trained on the "
        "INSTANCE 2022 challenge data reached a per-patient Dice of 0.843 on this dataset as an external test "
        "[@zheng2025]. The INSTANCE challenge itself, with 100 densely annotated training scans, was won with an "
        "average Dice of 0.6925 [@li2023instance]. Multiclass segmentation of traumatic lesions has been "
        "validated across centres [@monteiro2020], and three-dimensional networks have been applied to traumatic "
        "haemorrhage [@inkeaw2022]; on brain CT, three-dimensional architectures are not uniformly better than "
        "two-dimensional ones [@srikrishna2022].")
    W.p("Small datasets make evaluation fragile. Slices from one scan share anatomy, so slice-level partitioning "
        "leaks information between training and test sets [@kapoor2023;@varoquaux2022]; repeated tuning on one "
        "validation split overfits it [@varoquaux2022]; and metrics averaged over background and lesion, or over "
        "slices of very different lesion size, can misrepresent performance [@maierhein2024;@reinke2024]. "
        "Claims of architectural superiority often fail when baselines are trained rigorously [@isensee2024]. "
        "Data augmentation is the usual remedy for small datasets [@shorten2019;@chlap2021], but augmentation "
        "studies typically change several transforms at once, which confounds the amount of augmentation with "
        "the kind.")
    W.p("We therefore asked five questions under a single, pre-registered, scan-level protocol: how well a 2D "
        "U-Net segments traumatic haemorrhage on a held-out test set and on a second public dataset; how "
        "performance changes as each augmentation family is intensified while everything else is held fixed; "
        "whether undersampling of haemorrhage-free slices helps compared with keeping them and re-weighting; how "
        "much the CT window matters; and how performance varies across haemorrhage subtypes and lesion sizes.")

    # ---------------------------------------------------------------- methods
    W.h1("Methods")
    W.h2("Data")
    W.p(f"The primary dataset was PhysioNet CT-ICH version 1.3.1 [@physionet;@goldberger2000;@hssayeni2020]: "
        f"{v('derived.pn_scans', 'int')} non-contrast head CT scans of patients with traumatic brain injury, with "
        "approximately 5 mm axial slices, binary haemorrhage masks drawn by radiologists, and per-slice labels for "
        "five subtypes (epidural, subdural, subarachnoid, intraventricular and intraparenchymal) and fracture. "
        "Throughout, a scan is one CT volume from one patient and a slice is one axial image of a scan; "
        "statistics state which unit they refer to. The second dataset was BHSD [@wu2023bhsd], "
        f"{v('derived.bh_scans', 'int')} head CT volumes with voxel-level labels for the same five subtypes, "
        "distributed with an official training and test split. {tab:partitions} gives the number of scans, "
        "slices and lesion-positive slices in every partition, {tab:subtypes} the subtype distribution and "
        "lesion areas, and {tab:balance} the slice class distribution before and after each sampling "
        "scheme; {fig:overlays} shows representative slices. Patient demographics are given in Supplementary "
        "Table S1 with every figure labelled by group.")
    W.h2("Scan-level partitioning and leakage control")
    W.p("Every partition was made at the scan level and written to disk once, before any model was trained. "
        "Fifteen PhysioNet scans, stratified by haemorrhage status, formed a held-out test set; the remaining "
        "scans were divided into five development folds with stratified group k-fold cross-validation, so all "
        "slices of a scan always fell in the same fold. BHSD scans were assigned to training or test according to "
        "the dataset's official split, and BHSD training scans were divided into five folds that were paired with "
        f"the PhysioNet folds. An automated audit confirmed that no scan identifier occurred in two partitions or "
        f"two folds (maximum shared scans {v('data.leakage_max_shared', 'int')}) and compared low-resolution "
        f"fingerprints of every pair of scans across partitions to detect duplicated acquisitions (maximum "
        f"similarity {v('data.max_cross_partition_similarity', 'f3')}). Negative-slice undersampling, when used, "
        "was applied only to the training folds after partitioning, so no slice ever moved between partitions. "
        "The test sets were not used for any decision and were evaluated once.")
    W.h2("Hounsfield-unit processing and input representation")
    W.p("Volumes were read in Hounsfield units (HU) [@denotter], with the NIfTI scaling applied, values clipped "
        "to -1024 to 3071 HU, and a check that air and brain fell in their expected ranges. Each volume was "
        "reoriented to a canonical orientation using its affine, and per-slice labels were remapped with the "
        "slices. The left-right axis was then verified on the pixels with a mirror-symmetry test, so that "
        "horizontal flipping is an anatomically valid left-right mirror. Each slice was presented to the "
        "network as a square 230 mm field of view centred on the head, resampled to 320 x 320 pixels "
        "(0.72 mm per pixel), so all scans were seen at the same physical scale. Geometric augmentation was "
        "composed into the same affine transformation, so each image was interpolated once. The default input "
        "stacked three windows as channels (level/width 40/80, 80/200 and 600/2800 HU, for brain, subdural "
        "blood and bone). The windowing ablation compared this stack with single windows of 40/80, 40/120 (the "
        "window used in the original code) and 30/120 HU (the window stated in the original manuscript), and "
        "with the nnU-Net scheme of clipping to the 0.5-99.5 percentiles of lesion intensities and z-scoring "
        "[@isensee2021]. Predictions were mapped back to the native pixel grid with the exact inverse transform "
        "before any metric was computed.")
    W.h2("Segmentation models")
    W.p("The headline model was nnU-Net in its 2D configuration [@isensee2021], a 2D U-Net whose preprocessing, "
        "patch size and network depth are derived automatically from the data. It was trained on PhysioNet "
        "development scans together with BHSD training scans, and separately on PhysioNet development scans "
        "only, each with the five folds described above imposed on nnU-Net. Ablations used a 2D U-Net "
        "[@ronneberger2015] with an ImageNet-pretrained ResNet-34 encoder [@he2016] from the "
        "segmentation_models_pytorch library [@smp]. The original manuscript's network was ported layer for "
        "layer ({tab:architecture}): four encoder levels of 32, 64, 128 and 256 filters, each a single "
        "3 x 3 convolution followed by ReLU, batch normalisation and dropout (rate 0.01); a densely connected "
        "bridge of four such blocks with 512 filters each whose outputs are concatenated (described in the "
        "original manuscript as a dense layer, which it is not); nearest-neighbour upsampling with skip "
        "concatenation; and a 1 x 1 sigmoid output. It has 25,749,633 trainable parameters and 6,016 batch "
        "normalisation statistics.")
    W.h2("Training procedure")
    W.p("Every configuration was trained on a fixed schedule with no early stopping and no checkpoint "
        "selection: the weights at the final epoch are the model, so no validation result influenced which "
        "weights were evaluated. The ablation U-Net used AdamW (learning rate 3 x 10⁻⁴, weight decay 10⁻⁴), a "
        "linear warm-up over 2 epochs followed by cosine decay to zero, 50 epochs of 800 sampled slices "
        "(2,500 iterations at batch size 16), gradient clipping at norm 1 and bfloat16 mixed precision. The "
        "original recipe used Adam with learning rate 0.003718 (the value selected by the original Optuna "
        "search), the original augmentation, undersampling of negative slices to a 1:1 ratio and a Dice "
        "smoothing constant of 15, with the same number of training samples. nnU-Net used its default "
        "optimiser and polynomial learning-rate decay for "
        f"{v('derived.nnunet_epochs_PB', 'int')} epochs of 250 iterations; the epoch count was fixed from a "
        "two-epoch timing run and written to the pre-registration before the first fold trained. All arms, the "
        "primary endpoint, the comparisons and the post-processing rule were recorded in a pre-registration "
        "file before training. The Optuna search of the original manuscript selected settings on a validation "
        "subset that overlapped the training data; none of its results are used here. Training and validation "
        "loss curves are shown in {fig:learning}.")
    W.h2("Loss function")
    W.p("The training loss combined binary cross-entropy and soft Dice:")
    W.p("L = λ · L_BCE + (1 − λ) · (1 − D),   L_BCE = −(1/N) Σᵢ [w₊ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)],   "
        "D = (2 Σᵢ pᵢ yᵢ + s) / (Σᵢ pᵢ + Σᵢ yᵢ + s)", italic=True, align="center")
    W.p("where pᵢ is the predicted probability and yᵢ the label of pixel i, sums run over all N pixels of the "
        "mini-batch (batch Dice), λ = 0.5, the smoothing constant s = 1 (15 in the original recipe) and the "
        "positive-class weight w₊ = 1 unless stated. The imbalance experiment also used w₊ = 10, a per-slice "
        "weight of 0.2 on haemorrhage-free slices, and a Tversky term "
        "T = (Σ pᵢyᵢ + s) / (Σ pᵢyᵢ + α Σ (1 − pᵢ)yᵢ + β Σ pᵢ(1 − yᵢ) + s) with α = 0.7 on false negatives "
        "and β = 0.3 on false positives in place of D [@salehi2017;@yeung2022;@eelbode2020]. Because the loss "
        "value depends on λ, s and the class balance, it is not used as a performance measure in this revision.")
    W.h2("Class-imbalance handling")
    W.p("Haemorrhage occupies a small fraction of slices and of pixels. Six strategies were compared on the same "
        "recipe: random undersampling of negative slices to a 1:1 ratio (the original approach), sampling every "
        "slice uniformly, balanced sampling that keeps every negative slice available but draws positive and "
        "negative slices equally often (the default), and uniform sampling combined with a reduced loss weight "
        "for negative slices, a positive-pixel weight in the cross-entropy, or a Tversky loss. {tab:balance} "
        "reports how many negative slices each scheme discards.")
    W.h2("Augmentation-intensity experiment")
    W.p("To separate how much augmentation is applied from which augmentation is applied (reviewer-requested "
        "single-factor design), each transform family was intensified on its own while every other setting was "
        "held fixed. Starting from left-right flipping only, one family was added at 0.5, 1, 2 or 4 times its "
        "magnitude in Table 1 of the original manuscript: affine (rotation ±10°, scaling ±5 %, translation ±5 % "
        "of the field of view and shear ±10° at magnitude 1), elastic grid distortion (5 steps, distortion limit "
        "0.3), intensity (brightness ±0.10 and contrast ±0.109 of a 120 HU window applied in HU before "
        "windowing, and gamma 0.9-1.1) and Gaussian noise (variance 5-10 in 8-bit units of the 120 HU window, "
        "a standard deviation of 1.05-1.49 HU). No augmentation, and the original moderate and extreme policies "
        "reproduced verbatim, were included as reference arms; these four arms were repeated with three random "
        "seeds. To test whether a policy preserves haemorrhage morphology, every policy was applied to "
        "development slices and compared with the un-augmented rendering: the ratio of lesion areas, the shift "
        "in mean lesion intensity, the fraction of lesion pixels remaining in the 40-100 HU range of acute blood, "
        "and the change in boundary compactness.")
    W.h2("Post-processing")
    W.p("For each evaluated model, the probability threshold (0.10-0.90), a minimum three-dimensional component "
        "volume (0-200 mm³) and an optional brain-mask constraint were chosen on out-of-fold development "
        "predictions only, by maximising the mean per-scan score over all development scans, where a "
        "haemorrhage-positive scan scores its volumetric Dice and a haemorrhage-negative scan scores 1 if "
        "nothing is predicted and 0 otherwise. A nested leave-one-fold-out estimate quantifies how much of the "
        "gain is selection optimism. The selected operating point was frozen before the test sets were opened. "
        "Results at the fixed threshold of 0.5 without post-processing are also reported.")
    W.h2("Evaluation metrics")
    W.p("All metrics were computed for the lesion class only, on the native voxel grid, from binary masks "
        "obtained at the stated threshold [@taha2015;@maierhein2024;@reinke2024]. The primary metric is the "
        "volumetric Dice coefficient of each haemorrhage-positive scan, averaged over scans. Secondary metrics "
        "are pooled-voxel Dice and IoU, precision, recall (sensitivity) and specificity; per-slice Dice and IoU "
        "averaged over lesion-positive slices; lesion-wise sensitivity, where a ground-truth connected component "
        "counts as detected when at least 10 % of its voxels are predicted; false-positive components per scan; "
        "the 95th-percentile Hausdorff distance and average symmetric surface distance in millimetres and the "
        "normalised surface Dice at 2 mm; and scan-level and slice-level detection (sensitivity and specificity "
        "of predicting any haemorrhage, and the area under the ROC curve of the maximum predicted probability). "
        "Dice is undefined for scans or slices with an empty reference and is not averaged there; false positives "
        "on those scans are counted by specificity, false-positive components and the detection metrics. The "
        "binary IoU reported in the original manuscript was computed with tf.keras.metrics.BinaryIoU "
        "[@tfbinaryiou], whose default averages the IoU of the background and lesion classes; it is reported once "
        "for contrast and is not a lesion metric.")
    W.h2("Statistical analysis")
    W.p("Cross-validation comparisons were paired on matched (seed, fold) blocks with the Wilcoxon signed-rank "
        "test [@wilcoxon1945], Holm-corrected within each experiment [@holm1979], with Cliff's delta as effect "
        "size [@cliff1993] and a paired bootstrap 95 % confidence interval of the mean difference. Monotone "
        "dose-response across augmentation magnitudes was tested with Page's L test [@page1963]. Comparisons "
        "with too few blocks to reach p < 0.05 are flagged and reported descriptively, not as evidence of no "
        "effect. Test-set confidence intervals were obtained by bootstrapping scans, so that slices of one scan "
        "are resampled together.")
    W.h2("Implementation and computation")
    W.p("Experiments ran in PyTorch on a single NVIDIA L4 GPU in Google Colab. Parameters, forward-pass "
        "operations per slice, training time, inference time per slice and peak GPU memory were logged for "
        "every run. All code, the pre-registration and the scan-level splits are "
        "available from the authors' repository.")

    # ---------------------------------------------------------------- results
    W.h1("Results")
    W.h2("Cohort")
    _table_partitions(W)
    _table_subtypes(W)
    _table_balance(W)
    W.figure("overlays", "audit_overlays.png", "Representative lesion-positive slices with the reference "
             "haemorrhage outline (red), displayed with a 40/80 HU window after reorientation.")
    _table_architecture(W)
    W.figure("learning", "fig_learning_curves.png", "Training and validation loss (mean over the five "
             "scan-level folds) for the original model and recipe and for the flip-only U-Net. Training ran for a "
             "fixed number of epochs; the weights at the final epoch (dotted line) are the model.")
    W.h2("Held-out test performance")
    W.p(f"{{tab:test}} reports every model on both test sets. The headline nnU-Net model reached a "
        f"volumetric Dice of {v(f'test.{HEAD}.physionet_recipe.dice_vol_mean')} (95 % CI "
        f"{v(f'test.{HEAD}.physionet_recipe.dice_vol_mean_ci95', 'ci')}) on the "
        f"{v(f'test.{HEAD}.physionet_recipe.n_pos_scans', 'int')} haemorrhage-positive PhysioNet test scans, a mean "
        f"per-slice Dice of {v(f'test.{HEAD}.physionet_recipe.dice_slice_mean')} and per-slice lesion IoU of "
        f"{v(f'test.{HEAD}.physionet_recipe.iou_slice_mean')}, precision "
        f"{v(f'test.{HEAD}.physionet_recipe.precision_micro')} and recall "
        f"{v(f'test.{HEAD}.physionet_recipe.recall_micro')}, a median HD95 of "
        f"{v(f'test.{HEAD}.physionet_recipe.hd95_mm_median', 'f1')} mm, and identified haemorrhage-positive scans "
        f"with sensitivity {v(f'test.{HEAD}.physionet_recipe.scan_sensitivity')} and specificity "
        f"{v(f'test.{HEAD}.physionet_recipe.scan_specificity')} (AUC "
        f"{v(f'test.{HEAD}.physionet_recipe.scan_auc')}). On the BHSD test scans its volumetric Dice was "
        f"{v(f'test.{HEAD}.bhsd_recipe.dice_vol_mean')} (95 % CI {v(f'test.{HEAD}.bhsd_recipe.dice_vol_mean_ci95', 'ci')}). "
        f"Trained on PhysioNet alone, nnU-Net reached {v(f'test.{PONLY}.physionet_recipe.dice_vol_mean')} on the "
        f"PhysioNet test scans and {v(f'test.{PONLY}.bhsd_recipe.dice_vol_mean')} on BHSD, which it had never "
        "seen. The post-processing recipe and its nested development estimate are given in the table note, and "
        "out-of-fold development results for both nnU-Net models are in {tab:nnunet_cv}.")
    _table_test(W)
    _table_nnunet_cv(W)
    W.h2("Class-imbalance handling")
    W.p("{tab:imbalance} compares the six imbalance strategies against balanced sampling. "
        + (_cmp_sentence(S, "E2_imbalance", lambda a: a.replace("_", " ")) or f"{PEND}[pending: E2 results]{PEND}"))
    _table_cv(W, "imbalance", "E2_imbalance", "Class-imbalance strategies (five-fold scan-level cross-validation on "
              "PhysioNet development scans; mean over folds). Differences are paired against balanced sampling.")
    W.h2("CT window")
    W.p("{tab:window} compares windows. "
        + (_cmp_sentence(S, "E3_window", lambda a: a.replace("_", " ")) or f"{PEND}[pending: E3 results]{PEND}"))
    _table_cv(W, "window", "E3_window", "CT window and intensity normalisation (five-fold scan-level "
              "cross-validation). Differences are paired against the three-window stack.")
    W.h2("Augmentation intensity")
    fr = ((S.get("cv") or {}).get("E4_augmentation") or {}).get("friedman_composites") or {}
    fr_txt = (f"Across no augmentation, flip only, and the original moderate and extreme policies, the Friedman "
              f"test gave chi² = {fr['chi2']:.2f}, {fmt_value(fr['p'], 'p')} over {fr['n_blocks']} matched blocks. "
              if fr.get("p") is not None and not (isinstance(fr.get("p"), float) and math.isnan(fr["p"])) else "")
    W.p("{fig:dose} shows volumetric Dice as each family is intensified, and Tables {tabn:augmentation} and "
        "{tabn:trend} give the paired comparisons and trend tests. " + fr_txt + (_trend_sentence(S) or "")
        + " " + (_cmp_sentence(S, "E4_augmentation", lambda a: a) or f"{PEND}[pending: E4 results]{PEND}")
        + " {tab:morphology} reports how each policy changes lesion area, intensity and boundary shape.")
    W.figure("dose", "fig_dose_response.png", "Volumetric Dice (mean over scan-level folds, shaded 95 % "
             "interval) as one augmentation family is intensified from flipping only (0) to four times its "
             "magnitude in the original Table 1; all other settings fixed. The dashed line is no augmentation.")
    _table_cv(W, "augmentation", "E4_augmentation", "Augmentation policies (five-fold scan-level "
              "cross-validation; no augmentation, flip only and the two original policies with three seeds). "
              "Differences are paired against flip only.")
    _table_trend(W)
    _table_morphology(W)
    W.h2("Architecture and computation")
    W.p("{tab:compute} lists the encoders compared and their computational cost. "
        + (_cmp_sentence(S, "E5_architecture", lambda a: a.replace("_", " ")) or f"{PEND}[pending: E5 results]{PEND}"))
    _table_compute(W)
    W.h2("Haemorrhage subtypes and lesion size")
    W.p("{tab:subtype_test} reports performance by subtype on both test sets; subtype labels are per "
        "slice in PhysioNet and per voxel in BHSD, so per-subtype lesion sensitivity is available for BHSD only. "
        "{tab:size_test} stratifies performance by lesion size. Strata with few slices are reported with "
        "their counts so that their uncertainty is visible.")
    _table_subtype_test(W)
    _table_size_test(W)
    W.h2("Failure analysis")
    W.p(f"On the PhysioNet test scans, {v(f'test.{HEAD}.physionet_fp_near_bone_fraction', 'pct')} of the "
        "headline model's false-positive components lay mostly within 3 mm of bone, the partial-volume pattern "
        "described in the original manuscript. {fig:qualitative} shows CT, reference, prediction and an error "
        "overlay for the best slices, boundary errors, missed lesions and false positives, and {tab:failure} "
        "compares the error profile of every augmentation regime.")
    W.figure("qualitative", f"fig_qualitative_{HEAD}.png", "Held-out PhysioNet test slices: CT (40/80 HU), "
             "reference mask, prediction and overlay of true positives (aqua), false positives (orange) and "
             "false negatives (blue), for the best slices, boundary errors, missed lesions and false positives.")
    _table_failure(W)
    W.h2("Relation to the result reported in the original manuscript")
    W.p(f"The original manuscript reported a validation binary IoU of about 0.85. That figure combined three "
        "problems: the validation slices were a subset of the slices the model was trained on, the split was "
        "made at slice level, and the metric averaged background IoU with lesion IoU. Under the present "
        f"protocol the same network and recipe reach a mean per-slice lesion IoU of "
        f"{v('test.paper.physionet_recipe.iou_slice_mean')} on held-out scans, while the background-averaged "
        f"metric on the same predictions is {v('test.paper.physionet_recipe.binary_iou_classavg_ALLSLICES')}.")

    # ---------------------------------------------------------------- discussion
    W.h1("Discussion")
    W.p(_data_sentence(S) + " A model trained on the INSTANCE data reached a per-patient Dice of 0.843 on "
        "this dataset [@zheng2025], and label-efficient methods benefit from additional data [@spahr2023]; both "
        "point to training data, rather than architecture, as the main constraint at this scale [@isensee2024]. "
        "Performance on the external dataset indicates how much of the in-domain accuracy transfers to scans "
        "from another source.")
    W.p("The augmentation experiment separates the kind of augmentation from its amount. Because only one family "
        "changes at a time, a decline at high magnitude can be attributed to that family. We offer, as a "
        "hypothesis rather than a demonstrated mechanism, that transforms which move lesion intensities out of "
        "the range of acute blood or deform thin extra-axial collections produce training examples that do not "
        "resemble real haemorrhage; the morphology measurements in {tab:morphology} quantify how far each "
        "policy moves lesions in intensity, area and boundary shape, and should be read together with the "
        "performance results. The original manuscript attributed the failure of its extreme policy to departures "
        "from a low-dimensional anatomical manifold and to conflicting gradients; neither was measured, and we "
        "no longer make either claim.")
    W.p("Undersampling negative slices, the strategy used originally, discards most of the haemorrhage-free "
        "anatomy ({tab:balance}); the imbalance experiment shows whether keeping those slices and "
        "balancing batches or re-weighting the loss changes accuracy and false positives. The windowing "
        "ablation answers whether the choice of window, a question raised about the original 30/120 HU "
        "description, matters for this task.")
    W.p("Performance depends strongly on lesion size and subtype (Tables {tabn:subtype_test} and {tabn:size_test}). "
        "A single pooled average says little about small haemorrhages, which are the hardest to see and among "
        "the most important not to miss. Pooling five traumatic subtypes into one binary mask was necessary "
        "because PhysioNet provides pixel masks without subtype identity; the stratified results show where "
        "that pooling hides weaker performance. Thresholds such as 0.5 or 0.7 IoU have no established meaning "
        "for haemorrhage segmentation and are not used to grade results here.")

    W.h1("Limitations")
    W.p("The primary dataset contains 75 scans from a single source and scanner population, with young patients "
        "and thick slices; the BHSD dataset provides a second source but is not a prospective clinical sample. "
        "Fifteen test scans give wide confidence intervals. Annotations were made on thick slices and resampled, "
        "which limits achievable boundary accuracy for thin collections. The models are two-dimensional and do "
        "not use neighbouring slices. Subtype analyses on PhysioNet rely on slice-level labels. Generalisation to "
        "other scanners, acquisition protocols and patient populations, and any effect on clinical outcomes or "
        "workflow, were not assessed and would require prospective clinical validation.")

    W.h1("Conclusion")
    W.p("We re-evaluated 2D U-Net segmentation of traumatic intracranial haemorrhage under scan-level validation "
        "with a held-out and an external test set, lesion-only metrics, single-factor augmentation experiments "
        "and paired statistics. The results quantify segmentation and detection capability on retrospective "
        "public data. They do not establish clinical decision-making capability: a segmentation model with this "
        "level of accuracy may support research on haemorrhage volume or triage, but it is not a substitute for "
        "radiological interpretation and has not been validated in a clinical workflow.")

    W.h1("Acknowledgements")
    W.p("We thank Inspirit AI for mentorship, and the creators of the PhysioNet CT-ICH and BHSD datasets for "
        "making their annotated data public.")
    W.h1("Data and code availability")
    W.p("PhysioNet CT-ICH and BHSD are publicly available from their providers. The code, pre-registration, "
        "scan-level splits and all result tables are available from the authors' repository.")
    W.references()
    _supplementary(W)
    W.save(out)
    return W


def _supplementary(W):
    W.h1("Supplementary Material")
    rows = [[r.get("group"), r.get("patients"), r.get("age mean"), r.get("age SD"), r.get("age min"), r.get("age max"),
             r.get("male"), r.get("female")] for r in ((W.S.get("data") or {}).get("demographics") or [])]
    W.p("Supplementary Table S1. Demographics of the PhysioNet CT-ICH patients by haemorrhage status (ICH-positive: "
        "at least one slice with a haemorrhage mask). Age in years.", bold=True)
    if not rows:
        W.p(f"{PEND}[pending: demographics table]{PEND}")
        return
    t = W.doc.add_table(rows=1 + len(rows), cols=8)
    t.style = W.doc.styles["Table Grid"]
    for j, h in enumerate(["group", "patients", "mean age", "SD", "min", "max", "male", "female"]):
        t.rows[0].cells[j].text = h
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            t.rows[i].cells[j].text = _cell(val)


# ----------------------------------------------------------------------------- tables
def _recs(S, *path):
    cur = S
    for k in path:
        cur = (cur or {}).get(k) if isinstance(cur, dict) else None
    return cur or []


def _table_partitions(W):
    rows = [[r["partition"].replace("pn_", "PhysioNet ").replace("bh_", "BHSD "), r["scans"], r["ICH-positive scans"],
             r["slices"], r["lesion-positive slices"], r["lesion-negative slices"]]
            for r in _recs(W.S, "data", "partitions")]
    W.table("partitions", "Scans and slices in every partition (scans and slices are counted separately).",
            ["partition", "scans", "ICH-positive scans", "slices", "lesion-positive slices", "lesion-negative slices"], rows)


def _table_subtypes(W):
    rows = [[r["dataset"], r["subtype"], r["slices"], r["scans"], r.get("lesion area mm2 (median)"),
             r.get("lesion area mm2 (IQR)")] for r in _recs(W.S, "data", "subtypes")]
    W.table("subtypes", "Haemorrhage subtypes: lesion-positive slices and scans containing each subtype, and "
            "lesion area per slice (mm²). A slice can contain more than one subtype.",
            ["dataset", "subtype", "slices", "scans", "median area mm²", "IQR mm²"], rows)


def _table_balance(W):
    rows = [[r["scheme"], r["slices_total"], r["pos_slices"], r["neg_slices"], r["neg_slices_ever_used"],
             r["neg_discarded_fraction"], r["expected_pos_fraction_per_batch"]]
            for r in _recs(W.S, "data", "class_balance")]
    W.table("balance", "Class distribution of development training slices before and after each sampling scheme "
            "(mean over the five folds).",
            ["scheme", "slices", "positive", "negative", "negatives used", "fraction discarded", "positive share per batch"], rows)


def _table_architecture(W):
    rows = [["encoder 1", "conv 3x3, ReLU, BN, dropout 0.01", 32, "256x256"],
            ["encoder 2", "max-pool 2x2; conv 3x3, ReLU, BN, dropout", 64, "128x128"],
            ["encoder 3", "max-pool 2x2; conv 3x3, ReLU, BN, dropout", 128, "64x64"],
            ["encoder 4", "max-pool 2x2; conv 3x3, ReLU, BN, dropout", 256, "32x32"],
            ["bridge", "max-pool 2x2; 4 densely connected blocks (concatenated)", "4 x 512 (2,304 out)", "16x16"],
            ["decoder 4-1", "nearest upsample x2, concat skip, conv 3x3, ReLU, BN, dropout", "256/128/64/32", "32 to 256"],
            ["output", "conv 1x1, sigmoid", 1, "256x256"],
            ["parameters", "25,749,633 trainable + 6,016 BN statistics", "", ""]]
    W.table("architecture", "The original manuscript's U-Net (input 256 x 256 x 1).",
            ["stage", "operations", "filters", "resolution"], rows)


TEST_COLS = [("dice_vol_mean", "vol. Dice"), ("dice_slice_mean", "slice Dice"), ("iou_slice_mean", "slice IoU"),
             ("precision_micro", "prec."), ("recall_micro", "recall"), ("lesion_sensitivity", "lesion sens."),
             ("fp_lesions_per_scan", "FP/ scan"), ("scan_auc", "scan AUC")]
TEST_COLS_2 = [("dice_vol_mean_ci95", "vol. Dice 95 % CI"), ("hd95_mm_median", "HD95 mm (median)"),
               ("assd_mm_mean", "ASSD mm"), ("nsd_mean", "NSD 2 mm"), ("scan_sensitivity", "scan sens."),
               ("scan_specificity", "scan spec."), ("specificity_micro", "voxel spec.")]
MODEL_LABEL = {"nnU-Net P+B": "nnU-Net, P+B", "nnU-Net P": "nnU-Net, P only",
               "paper": "original U-Net", "aug_none": "U-Net, no aug.",
               "aug_base": "U-Net, flip", "aug_moderate": "U-Net, moderate",
               "aug_extreme": "U-Net, extreme"}


def _table_test(W):
    rows, rows2 = [], []
    for m, lab in MODEL_LABEL.items():
        for part in ("physionet", "bhsd"):
            r = _test_row(W.S, m, part)
            if r:
                name = f"{lab} / {'PhysioNet' if part == 'physionet' else 'BHSD'}"
                rows.append([name] + [r.get(k) for k, _ in TEST_COLS])
                rows2.append([name] + [r.get(k) for k, _ in TEST_COLS_2])
    recipe = (W.S.get("postproc") or {}).get(HEAD, {})
    note = None
    if recipe:
        rc = recipe.get("recipe", {})
        note = (f"Headline operating point chosen on development scans: threshold {rc.get('threshold')}, minimum "
                f"component {rc.get('min_mm3')} mm³, brain mask {'on' if rc.get('brain_mask') else 'off'}; nested "
                f"development score {recipe.get('nested_score', float('nan')):.3f} versus "
                f"{recipe.get('default_score', float('nan')):.3f} at threshold 0.5 without post-processing.")
    W.table("test", "Held-out test results, each test set evaluated once (model / test set; P+B = trained on "
            "PhysioNet and BHSD, P = PhysioNet only). Volumetric Dice is averaged over haemorrhage-positive scans; "
            "slice metrics over lesion-positive slices; precision and recall pool all voxels; FP/scan counts "
            "false-positive components. Boundary and detection metrics for the same predictions follow in the "
            "second part of the table.", ["model / test set"] + [c for _, c in TEST_COLS], rows, note)
    if rows2:
        W.table_continued(["model / test set"] + [c for _, c in TEST_COLS_2], rows2)


def _table_nnunet_cv(W):
    rows = []
    for which, lab in (("P+B", "PhysioNet+BHSD"), ("P", "PhysioNet only")):
        t = (W.S.get("nnunet_cv") or {}).get(which)
        if t:
            mu, sd = t["mean"], t["sd"]
            rows.append([lab, t.get("folds")] + [f"{mu.get(k, float('nan')):.3f} ± {sd.get(k, float('nan')):.3f}"
                                                 for k in ("dice_vol_mean", "dice_slice_mean", "scan_auc", "bhsd_dice_vol_mean")])
    W.table("nnunet_cv", "nnU-Net 2D out-of-fold development results (mean ± SD over folds, threshold 0.5).",
            ["training data", "folds", "PhysioNet volumetric Dice", "PhysioNet slice Dice", "PhysioNet scan AUC",
             "BHSD volumetric Dice"], rows)


def _table_cv(W, key, exp, caption):
    summ = _recs(W.S, "cv", exp, "summary")
    tests = {t["A"]: t for t in _recs(W.S, "cv", exp, "tests")}
    rows = []
    for r in summ:
        t = tests.get(r["arm"], {})
        mu, sd = r.get("mean"), r.get("std")
        ok = isinstance(sd, (int, float)) and not math.isnan(sd)
        rows.append([r["arm"], r.get("count"), f"{mu:.3f} ± {sd:.3f}" if ok else mu,
                     r.get("dice_slice_mean"), r.get("fp_lesions_per_scan"),
                     t.get("mean_diff"), t.get("p_holm"), t.get("cliffs_delta")])
    W.table(key, caption + " Precision and recall per arm are in the supplementary result tables.",
            ["arm", "blocks", "vol. Dice (mean ± SD)", "slice Dice", "FP/scan", "difference vs reference",
             "Holm p", "Cliff's delta"], rows)


def _table_trend(W):
    rows = [[r["family"], r.get("n_blocks"), r.get("L"), r.get("p")] for r in _recs(W.S, "cv", "E4_augmentation", "trend")]
    W.table("trend", "Page's L test for a monotone decrease of volumetric Dice across magnitudes 0, 0.5, 1, 2 and 4 "
            "within each augmentation family.", ["family", "blocks", "L", "p"], rows)


def _table_morphology(W):
    rows = [[r["arm"], r.get("area_ratio_mean"), r.get("lesion_hu_shift_mean_abs"),
             r.get("frac_lesion_px_in_blood_range_after"), r.get("compactness_ratio"), r.get("lesion_lost_rate")]
            for r in (W.S.get("morphology") or [])]
    W.table("morphology", "Effect of each augmentation policy on haemorrhage morphology in development slices "
            "(augmented versus un-augmented rendering of the same slice).",
            ["policy", "area ratio", "|mean HU shift|", "lesion pixels in 40-100 HU", "compactness ratio",
             "lesion lost"], rows)


def _table_compute(W):
    rows = [[r["arm"], r.get("params_M"), r.get("GFLOPs/slice"), r.get("train_min"), r.get("inference_ms/slice"),
             r.get("peak_GPU_GB"), r.get("dice_vol_mean")] for r in (W.S.get("compute") or [])]
    W.table("compute", "Encoders and computational cost (single NVIDIA L4; training time per fold).",
            ["encoder", "parameters (M)", "GFLOPs per slice", "training min", "inference ms/slice", "peak GPU GB",
             "volumetric Dice"], rows)


def _strata(W, model, name):
    import pandas as pd
    p = W.figdir.parent / "tables" / f"test_{slug(model)}" / name
    return pd.read_csv(p) if p.exists() else None


def _table_subtype_test(W):
    rows = []
    for part in ("physionet", "bhsd"):
        t = _strata(W, "nnU-Net P+B", f"{part}_subtype_strata.csv")
        if t is None:
            continue
        for r in t.to_dict("records"):
            rows.append(["PhysioNet" if part == "physionet" else "BHSD", r["subtype"], r["pos_slices"], r["scans"],
                         r.get("dice_slice_mean"), r.get("slice_sensitivity"), r.get("voxel_recall_slices"),
                         r.get("lesion_sensitivity")])
    W.table("subtype_test", "Headline model by haemorrhage subtype on the test sets.",
            ["test set", "subtype", "slices", "scans", "slice Dice", "slice sensitivity", "voxel recall",
             "lesion sensitivity (BHSD)"], rows)


def _table_size_test(W):
    rows = []
    for part in ("physionet", "bhsd"):
        t = _strata(W, "nnU-Net P+B", f"{part}_size_strata.csv")
        if t is None:
            continue
        for r in t.to_dict("records"):
            rows.append(["PhysioNet" if part == "physionet" else "BHSD", r["stratum"], r["n"], r.get("dice_slice_mean"),
                         r.get("slice_sensitivity"), r.get("lesion_sensitivity")])
    W.table("size_test", "Headline model by lesion size on the test sets (slice area and 3D component volume).",
            ["test set", "stratum", "n", "slice Dice", "slice sensitivity", "lesion sensitivity"], rows)


def _table_failure(W):
    rows = []
    for m, lab in MODEL_LABEL.items():
        r = _test_row(W.S, m, "physionet")
        if r:
            rows.append([lab, r.get("fp_lesions_per_scan"), r.get("fp_lesions_per_neg_scan"),
                         (W.S["test"][slug(m)] or {}).get("physionet_fp_near_bone_fraction"),
                         r.get("n_pos_scans_empty_pred"), r.get("hd95_mm_median")])
    W.table("failure", "Error profile on the PhysioNet test scans by model and augmentation regime.",
            ["model", "FP components/scan", "FP components/negative scan", "share of FP near bone",
             "positive scans with no prediction", "median HD95 mm"], rows)


# ----------------------------------------------------------------------------- response letter
RESPONSES = [
    ("Major 1", "Remove section numbers, bullet points and numbered points.",
     "Done. The manuscript is generated with unnumbered bold headings and contains no bullets or numbered lists; "
     "Times New Roman 12 pt with single spacing throughout."),
    ("Major 2", "Partition at the patient/scan level, report scans per set, show no leakage, undersample inside training only, use patient-level k-fold CV with a held-out test set.",
     "We agree, and we found that the original validation slices were in fact a subset of the training slices. All "
     "partitions are now made at scan level and frozen before training: {derived.pn_test} PhysioNet test scans, "
     "{derived.pn_dev} development scans in five stratified group folds, and the BHSD official test split. "
     "Table 1 lists scans and slices per partition; an automated audit shows {data.leakage_max_shared} shared scans "
     "between any two partitions or folds and a maximum fingerprint similarity of "
     "{data.max_cross_partition_similarity} between scans in different partitions. Undersampling is applied only "
     "to training folds (Methods, Class-imbalance handling)."),
    ("Major 3", "No repeated runs, SDs or statistical tests.",
     "Every configuration now runs on five scan-level folds; the no-augmentation, flip-only, moderate and extreme "
     "policies run with three seeds (15 matched blocks). We report mean ± SD, paired Wilcoxon tests with Holm "
     "correction, Cliff's delta and paired bootstrap CIs, and flag comparisons that are under-powered "
     "(Statistical analysis; Tables 7-10)."),
    ("Major 4", "Augmentation effect not isolated; vary one family at graded magnitudes; correct the 4.2x arithmetic.",
     "We redesigned the experiment as requested: starting from left-right flipping only, each family (affine, "
     "elastic, intensity, noise) is added alone at 0.5, 1, 2 and 4 times its Table 1 magnitude, with Page's trend "
     "test per family (Figure 3, Tables 9-10). The arithmetic is corrected: 0.228/0.0524 = 4.35, i.e. a 335 % "
     "increase; loss values are no longer used as performance measures."),
    ("Major 5", "Quantify class distributions before/after undersampling and compare with class-wise losses.",
     "Table 3 gives positive/negative slice counts before and after each scheme and the fraction of negatives "
     "discarded. The imbalance experiment compares 1:1 undersampling, uniform sampling, balanced sampling, "
     "negative-slice down-weighting, positive-pixel weighting and a Tversky loss (Table 7)."),
    ("Major 6", "Give the loss formula and weights.",
     "The loss is now given as an equation with all constants (λ = 0.5, s = 1, or 15 in the original recipe; batch "
     "Dice; w₊ = 1 unless stated), together with the Tversky variant (Methods, Loss function)."),
    ("Major 7", "Optuna search, checkpoint selection and validation overfitting; report an untouched test set.",
     "All Optuna results are withdrawn: the search used validation slices that were also training slices. Every "
     "arm, the primary endpoint and the post-processing rule were written to a pre-registration file before "
     "training. Models are trained on a fixed schedule and the final epoch is the model, so no checkpoint is "
     "chosen on validation data. Test results appear in their own table (Table 5), each test set evaluated "
     "once."),
    ("Major 8", "IoU insufficient; report Dice, precision, recall, specificity, lesion sensitivity, boundary metrics, foreground IoU, metric constructor, threshold, empty masks, aggregation, patient-level sensitivity and FP per scan.",
     "All are now reported (Evaluation metrics, Table 5). The original metric was tf.keras.metrics.BinaryIoU with "
     "its default averaging of background and lesion IoU, at threshold 0.5, aggregated over all validation "
     "slices; this inflated the 0.85. We now report lesion-only metrics on the native grid, state the threshold "
     "and empty-mask rules, and give scan-level sensitivity and false-positive components per scan."),
    ("Major 9", "Report results per haemorrhage subtype.",
     "Table 2 gives subtype sample sizes and lesion areas; Table 13 gives subtype-specific Dice, sensitivity and "
     "voxel recall on both test sets, with lesion-level sensitivity per subtype on BHSD, whose labels are per "
     "voxel."),
    ("Major 10", "Generalisation claims not supported.",
     "The claims are removed. We now test generalisation directly: a model trained only on PhysioNet is evaluated "
     "on the external BHSD test set (Table 5), and the Limitations state what remains untested."),
    ("Major 11", "Intracranial vs intracerebral haemorrhage; choose and rewrite the framing.",
     "The paper is now explicitly about traumatic intracranial haemorrhage. The stroke epidemiology and its "
     "references are removed and replaced with TBI epidemiology; the Introduction states the distinction, and "
     "pooling of subtypes into one binary mask is justified (PhysioNet has no per-pixel subtype labels) and "
     "complemented by per-subtype results."),
    ("Major 12", "Extreme-augmentation failure mechanism not demonstrated; drop the manifold argument.",
     "The manifold and gradient explanations are removed. We state one hypothesis explicitly as a hypothesis and "
     "provide measurements of how each policy changes lesion intensity, area and boundary shape (Table 11)."),
    ("Major 13", "Expand the literature review to 25-30 recent peer-reviewed primary articles.",
     "The Introduction and Discussion now draw on {derived.n_refs} references, most of them peer-reviewed primary "
     "research on haemorrhage detection, segmentation, TBI imaging, evaluation methodology and augmentation."),
    ("Major 14", "Audit every citation against its claim.",
     "Every citation was checked against its source. Several were removed or replaced: the intracerebral "
     "haemorrhage epidemiology (wrong population), a COVID-19 classification paper (unrelated), and a clinical "
     "workflow paper that we had misrepresented (it piloted a framework on lymphoscintigraphy and reports neither "
     "ICH nor 93 % accuracy); its place is taken by studies of ICH worklist prioritisation. One subtype reference "
     "could not be verified and was replaced. The 2D-versus-3D citation is now restricted to what it shows "
     "(tissue classification)."),
    ("Recommended 1", "Numerical comparison of architectures.",
     "Table 12 compares the original U-Net with ResNet-34, DenseNet-121, VGG-16 and MobileNetV2 encoders and nnU-Net, "
     "with cost. The earlier classification models (CNN, VGG, InceptionV3 and others) addressed slice "
     "classification, a different task, and are no longer discussed."),
    ("Recommended 2", "Ablate the 30/120 HU window.",
     "Five windows are compared (Table 8), including 40/120 HU (what the original code used) and 30/120 HU (what "
     "the manuscript stated; the discrepancy is corrected)."),
    ("Recommended 3", "Characterise the five subtypes before the method.",
     "Table 2 (slices, scans and lesion areas per subtype and dataset)."),
    ("Recommended 4", "Relate lesion size to segmentation quality.",
     "Table 14 stratifies by slice lesion area and 3D component volume."),
    ("Recommended 5", "Scheduler and early-stopping details.",
     "There is no early stopping. The schedule (warm-up and cosine decay to zero, or nnU-Net's polynomial "
     "decay), its length and all optimiser settings are given in Training procedure."),
    ("Recommended 6", "Computational characteristics.",
     "Table 12: parameters, GFLOPs per slice, training time, inference time per slice and peak GPU memory."),
    ("Recommended 7", "Systematic failure-case analysis.",
     "Figure 4 shows best, boundary-error, missed and false-positive slices; Table 15 quantifies false positives "
     "per scan, the share near bone, missed scans and boundary error for every augmentation regime."),
    ("Recommended 8", "Show that augmentation preserves haemorrhage morphology.",
     "Table 11 measures, for every policy, the lesion area ratio, intensity shift, fraction of lesion pixels "
     "remaining in the 40-100 HU blood range and boundary compactness."),
    ("Recommended 9", "Gap between the best Optuna trial (0.044) and retraining (0.0524).",
     "Both numbers came from the leaked protocol and were validation losses, which are not comparable across "
     "settings; they are withdrawn. Learning curves for the current models are in Figure 2 with the final-epoch "
     "model marked; seeds are recorded per run."),
    ("Recommended 10", "Separate segmentation capability from clinical decision-making.",
     "The Conclusion now does so explicitly; 'reliable tool' and 'clinically meaningful' are removed."),
    ("Minor 1", "Distinguish scans from slices.", "Defined in Data and kept separate in every table."),
    ("Minor 2", "Class distribution after sampling.", "Table 3."),
    ("Minor 3", "Label the demographic statistics.",
     "Age and sex are now reported by group (all, ICH-positive, ICH-negative) with every figure labelled "
     "(Supplementary Material)."),
    ("Minor 4", "Architecture details.", "Table 4 and Segmentation models (filters, kernels, dropout, activations, parameters)."),
    ("Minor 5", "Loss formula.", "See Major 6."),
    ("Minor 6", "Binarisation threshold.",
     "Threshold 0.5 for all cross-validation results; the post-processing operating point is chosen on development "
     "data only and stated in Table 5."),
    ("Minor 7", "Consistent augmentation units.",
     "All parameters are now defined with units in the augmentation methods (degrees, fraction of FOV, HU, "
     "variance in 8-bit units converted to HU)."),
    ("Minor 8", "Which model produced which result.", "Withdrawn Optuna results; every reported number names its model and protocol."),
    ("Minor 9", "Mark the chosen checkpoint on training curves.", "Figure 2 marks the final epoch, which is the model."),
    ("Minor 10", "'Loss under 5.5 %'.", "Removed; losses are not percentages and are no longer reported as results."),
    ("Minor 11", "IoU thresholds 0.5/0.7.", "Removed; the Discussion states they have no established meaning for this task."),
    ("Minor 12", "Qualitative figures with CT, ground truth and prediction.", "Figure 4."),
    ("Minor 13", "Label, number and cite all figures and tables.",
     "Every table and figure is numbered in order of first citation, captioned and cited in the text."),
]


def build_response(S: dict, out: Path, n_refs: int):
    from docx import Document
    d = S.setdefault("derived", {})
    parts = {r["partition"]: r for r in (S.get("data") or {}).get("partitions", [])}
    d.update({"pn_test": parts.get("pn_test", {}).get("scans"), "pn_dev": parts.get("pn_dev", {}).get("scans"),
              "n_refs": n_refs})
    W = Writer(S, Path("."))
    W.p("Response to the reviewers", bold=True, align="center", size=14)
    W.p("Deep learning models for intracranial haemorrhage detection and masking (revised title: Segmentation of "
        "Traumatic Intracranial Haemorrhage on Head CT with 2D U-Nets: Augmentation Intensity, Class Imbalance and "
        "Generalisation under Scan-Level Validation)", align="center")
    W.p("We thank the reviewers for a thorough and constructive review. While addressing it we found that the "
        "original validation slices were a subset of the training slices and that the reported binary IoU "
        "averaged background and lesion. Every experiment has therefore been redone under a scan-level, "
        "pre-registered protocol with held-out and external test sets, and all results in the revised manuscript "
        "come from that protocol. Our responses follow the order of the review. Table and figure numbers refer to "
        "the revised manuscript.")
    for tag, comment, resp in RESPONSES:
        W.p(tag, bold=True)
        W.p(f"Comment (summary): {comment}", italic=True)
        filled = re.sub(r"\{([\w.\[\]=+@-]+)\}", lambda m: W.v(m.group(1), "int" if "scans" in m.group(1) or
                                                                 "shared" in m.group(1) or "pn_" in m.group(1) or
                                                                 "n_refs" in m.group(1) else "f3"), resp)
        W.p(f"Response: {filled}")
    W.save(out)
    return W


def build_all(paths, summary_path: Path | None = None) -> dict:
    summary_path = summary_path or (paths.root / "results_summary.json")
    S = json.loads(Path(summary_path).read_text()) if Path(summary_path).exists() else {}
    reg = paths.root / "preregistration.json"
    if reg.exists():
        ep = json.loads(reg.read_text()).get("nnunet", {}).get("epochs")
        if isinstance(ep, dict):
            S.setdefault("derived", {})["nnunet_epochs_PB"] = ep.get("P+B")
    out_dir = paths.root / "manuscript"
    W = build_manuscript(S, paths.figures, out_dir / "ICH_manuscript_revised.docx")
    R = build_response(S, out_dir / "Response_to_reviewers.docx", len(W.cited))
    pending = sorted(set(W.v.missing + R.v.missing))
    (out_dir / "pending_values.txt").write_text("\n".join(pending))
    return {"manuscript": str(out_dir / "ICH_manuscript_revised.docx"),
            "response": str(out_dir / "Response_to_reviewers.docx"),
            "references": len(W.cited), "pending_values": len(pending)}

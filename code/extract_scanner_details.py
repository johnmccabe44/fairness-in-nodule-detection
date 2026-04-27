import argparse
import csv
import zipfile
from io import BytesIO
from pathlib import Path

import pydicom
from tqdm import tqdm


def _to_scalar(value):
	if value is None:
		return ""
	if isinstance(value, (list, tuple)):
		return "\\".join(str(v) for v in value)
	return str(value)


def extract_details_from_zip(zip_path: Path) -> dict:
	row = {
		"zip_file": str(zip_path),
		"dicom_file": "",
		"manufacturer": "",
		"pixel_spacing": "",
		"slice_thickness": "",
		"error": "",
	}

	try:
		with zipfile.ZipFile(zip_path, "r") as zf:
			dcm_files = sorted(
				name for name in zf.namelist() if name.lower().endswith(".dcm")
			)

			if not dcm_files:
				row["error"] = "No .dcm file found"
				return row

			first_dcm = dcm_files[0]
			row["dicom_file"] = first_dcm

			with zf.open(first_dcm) as f:
				ds = pydicom.dcmread(BytesIO(f.read()), stop_before_pixels=True)

			row["manufacturer"] = _to_scalar(getattr(ds, "Manufacturer", ""))
			row["pixel_spacing"] = _to_scalar(getattr(ds, "PixelSpacing", ""))
			row["slice_thickness"] = _to_scalar(getattr(ds, "SliceThickness", ""))
	except Exception as e:
		row["error"] = str(e)

	return row


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Loop through ZIP files in a directory, read first DICOM header, "
			"and save scanner details to CSV."
		)
	)
	parser.add_argument("input_dir", type=Path, help="Directory containing .zip files")
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path("scanner_details.csv"),
		help="Output CSV path (default: scanner_details.csv)",
	)
	args = parser.parse_args()

	zip_files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() == ".zip")

	rows = [extract_details_from_zip(zp) for zp in tqdm(zip_files, desc="Processing ZIPs")]

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	with args.output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"zip_file",
				"dicom_file",
				"manufacturer",
				"pixel_spacing",
				"slice_thickness",
				"error",
			],
		)
		writer.writeheader()
		writer.writerows(rows)


if __name__ == "__main__":
	main()

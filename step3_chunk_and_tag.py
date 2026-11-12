import os
import re
import joblib

parsed_folder = "parsed_docs"


def has_real_visual_content(chunk_text, section_title, image_path_exists):
    if image_path_exists:
        return True
    t_low = chunk_text.lower()
    if section_title.strip().lower() == "page 1" and "visual figure" not in t_low and "| ---" not in chunk_text:
        return False
    if "visual figure" in t_low or "| ---" in chunk_text or "| :---" in chunk_text:
        return True
    if re.search(r"\b(figure|fig\.|table)\s+\d+", t_low):
        return True
    return False


def get_chunk_metadata(text, source_file, section_title, page_image_paths_str):
    text_lower = text.lower()

    materials = []
    if any(word in text_lower for word in ["steel", "martensite", "ferrite", "bainite", "austenite", "carburiz", "fe3c", "s35c", "4140", "4340", "axle", "wheel", "rail"]):
        materials.append("steel")
    if any(word in text_lower for word in ["aluminum", "aluminium", "2024", "7075"]):
        materials.append("aluminum")
    if any(word in text_lower for word in ["titanium", "ti-6al-4v"]):
        materials.append("titanium")
    if len(materials) == 0:
        materials.append("general_metal")

    failure_modes = []
    if any(word in text_lower for word in ["fatigue", "cyclic", "striation", "beach mark", "fish-eye", "endurance", "s-n"]):
        failure_modes.append("fatigue")
    if any(word in text_lower for word in ["corrosion", "pitting", "scc", "rust", "oxidation"]):
        failure_modes.append("corrosion")
    if any(word in text_lower for word in ["brittle", "cleavage", "intergranular", "embrittlement", "overload", "mvc", "weld"]):
        failure_modes.append("brittle_fracture")
    if any(word in text_lower for word in ["inclusion", "sulfide", "alumina", "astm e45", "porosity"]):
        failure_modes.append("inclusion_defect")
    if len(failure_modes) == 0:
        failure_modes.append("general_failure")

    has_img = bool(page_image_paths_str)
    valid_img = page_image_paths_str if has_real_visual_content(
        text, section_title, has_img) else None

    return {
        "source": source_file.replace(".md", ""),
        "section": section_title,
        "materials": materials,
        "failure_modes": failure_modes,
        "image_path": valid_img
    }


def extract_page_image_path(page_text):
    """Extracts one or more '|'-separated cropped figure paths from a page block."""
    for line in page_text.split("\n"):
        if "PAGE_SNAPSHOT:" in line or "FIGURE_IMAGE_PATH:" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                raw_val = parts[1].replace("]**", "").replace("]", "").strip()
                valid_paths = []
                for p in raw_val.split("|"):
                    cleaned = os.path.normpath(p.strip())
                    if os.path.exists(cleaned):
                        valid_paths.append(cleaned)
                if valid_paths:
                    return " | ".join(valid_paths)
    return None


def chunk_single_markdown_file(filename, start_chunk_id=0):
    filepath = os.path.join(parsed_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    if "\n## Page " in full_text:
        raw_blocks = full_text.split("\n## Page ")
        is_paged_pdf = True
    else:
        raw_blocks = full_text.split("\n## ")
        is_paged_pdf = False

    file_chunks = []
    chunk_id = start_chunk_id
    pending_short_text = ""

    for idx, block in enumerate(raw_blocks):
        block = block.strip()
        if len(block) < 30:
            continue

        if "vision extraction unavailable" in block.lower() or "blank or cover page" in block.lower():
            continue

        if is_paged_pdf and idx > 0:
            block = "Page " + block

        page_image_path = extract_page_image_path(block)

        lines = block.split("\n")
        section_title = lines[0].replace("#", "").strip()
        body_lines = [l.replace("**", "") if l.count("**") ==
                      1 else l for l in lines[1:] if "PAGE_SNAPSHOT:" not in l]
        body_joined = "\n".join(body_lines).strip()

        if len(body_joined) < 120 and not page_image_path:
            pending_short_text = body_joined
            continue

        if pending_short_text:
            body_lines = [pending_short_text, ""] + body_lines
            pending_short_text = ""

        total_body_chars = sum(len(l) for l in body_lines)
        chars_processed = 0
        current_lines = []
        current_length = 0

        for line in body_lines:
            current_lines.append(line)
            current_length += len(line)
            chars_processed += len(line)

            is_table_row = line.strip().startswith("|")
            remaining_chars = total_body_chars - chars_processed

            if current_length >= 1600 and remaining_chars >= 450 and not is_table_row:
                chunk_text = f"[{filename.replace('.md', '')} | {section_title}]\n" + "\n".join(
                    current_lines)
                meta = get_chunk_metadata(
                    chunk_text, filename, section_title, page_image_path)

                file_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": meta
                })
                chunk_id += 1

                current_lines = current_lines[-3:]
                current_length = sum(len(l) for l in current_lines)

        if current_length > 40:
            chunk_text = f"[{filename.replace('.md', '')} | {section_title}]\n" + "\n".join(
                current_lines)
            meta = get_chunk_metadata(
                chunk_text, filename, section_title, page_image_path)

            file_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": meta
            })
            chunk_id += 1

    return file_chunks


def create_chunks_from_markdown():
    all_chunks = []
    chunk_id = 0

    md_files = sorted([f for f in os.listdir(
        parsed_folder) if f.endswith(".md")])
    print(
        f"\n--- Splitting {len(md_files)} Documents Using Page- & Section-Aware Chunking ---")

    for filename in md_files:
        file_chunks = chunk_single_markdown_file(
            filename, start_chunk_id=chunk_id)
        all_chunks.extend(file_chunks)
        chunk_id += len(file_chunks)
        print(f" -> {filename}: created {len(file_chunks)} tagged chunks")

    joblib.dump(all_chunks, "chunks_data.pkl")
    print(
        f"\n=== STAGE 3 COMPLETE! Saved {len(all_chunks)} clean chunks to 'chunks_data.pkl' ===")
    return all_chunks


if __name__ == "__main__":
    create_chunks_from_markdown()

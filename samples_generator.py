import csv
import random
import re
import os
from offense_classifier_end_to_end.config import CATEGORIES, PLACEHOLDER_OPTIONS


def replace_placeholders(template: str) -> str:
    """
    Replaces any placeholder like {hit}, {light}, {place}, etc. found in the template.
    If a placeholder does not exist in placeholder_options, it is left as-is.
    """
    result = template

    # Identify all placeholders by scanning for curly braces
    # e.g. {hit}, {light}, {place}
    # You can do this more robustly with regex, but here's a simple approach:
    placeholders_found = []
    pattern = re.compile(r"{(.*?)}")
    placeholders_in_text = pattern.findall(template)

    for ph in placeholders_in_text:
        ph_list = PLACEHOLDER_OPTIONS.get(ph)
        if ph_list:
            replacement = random.choice(ph_list)
            # Replace all occurrences of this placeholder in the template
            result = result.replace("{" + ph + "}", replacement)

    return result


# Create augmented dataset
def create_dataset(output_file="datasets/samples.csv", samples_per_category=1500):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label"])
        total_desired_samples = len(CATEGORIES) * samples_per_category
        total_templates_count = 0
        for label, templates in CATEGORIES.items():
            for _ in range(len(templates)):
                total_templates_count += 1
        for label, templates in CATEGORIES.items():
            category_samples_count = int((len(templates) / total_templates_count) * total_desired_samples)
            print(f"Generating {category_samples_count} examples for {label}...")
            counter = 0
            while counter < category_samples_count:
                modulo = counter % len(templates)
                template = templates[modulo]
                text = replace_placeholders(template)
                if "}" in text or "{" in text:
                    print(f"Warning: Placeholder not replaced in template: {template}")
                writer.writerow([text, label])
                counter += 1
            # for _ in range(category_samples_count):
            #     template = random.choice(templates)
            #     text = replace_placeholders(template)
            #     if "}" in text or "{" in text:
            #         print(f"Warning: Placeholder not replaced in template: {template}")
            #     writer.writerow([text, label])


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    samples_csv_path = os.path.join(script_dir, "datasets/samples.csv")
    create_dataset(samples_csv_path, samples_per_category=1100)  # Adjust as needed

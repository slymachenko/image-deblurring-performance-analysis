export_env:
	@echo "Exporting environment with version information..."
	@conda env export --no-builds | grep -v "^prefix: " > _full_env_.yml
	@conda env export --from-history | grep -v "^prefix: "  > _min_env_.yml
	@awk ' \
		BEGIN { in_dep=0 } \
		/^dependencies:/ { in_dep=1; next } \
		/^prefix:/ { in_dep=0; next } \
		in_dep && $$0 ~ /^  - / { \
			pkg_line = $$0; sub(/^  - /, "", pkg_line); \
			split(pkg_line, parts, "="); \
			if (length(parts) >= 2) { \
				versions[parts[1]] = parts[2]; \
			} \
		} \
		END { \
			for (pkg in versions) { \
				print pkg" "versions[pkg]; \
			} \
		}' _full_env_.yml > version_map.txt
	@awk -v vmap="version_map.txt" ' \
		BEGIN { \
			while ((getline line < vmap) > 0) { \
				split(line, fields, " "); \
				if (length(fields) >= 2) { \
					versions[fields[1]] = fields[2]; \
				} \
			} \
			close(vmap); \
			in_dep=0; \
		} \
		{ \
			if ($$0 ~ /^dependencies:/) { \
				in_dep=1; \
				print; \
			} else if (in_dep && $$0 ~ /^  - /) { \
				pkg_line = $$0; sub(/^  - /, "", pkg_line); \
				# Remove any trailing version (if present) \
				gsub(/=.*/, "", pkg_line); \
				if (pkg_line in versions) { \
					print "  - " pkg_line"="versions[pkg_line]; \
				} else { \
					print $$0; \
				} \
			} else { \
				print $$0; \
			} \
		}' _min_env_.yml > environment.yml
	@rm -f _full_env_.yml _min_env_.yml version_map.txt
	@echo "Updated environment file with versions: environment.yml"
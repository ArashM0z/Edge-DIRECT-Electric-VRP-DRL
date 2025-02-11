# Workflows pending the `workflow` OAuth scope

These yamls are the intended `.github/workflows/` content. They are stored
here because the OAuth token used to push initial history lacked the
`workflow` scope.

To activate the CI/CD pipeline:

```bash
# 1. Refresh the local gh token with the workflow scope
gh auth refresh -h github.com -s workflow

# 2. Move them into place and push
git mv .github/_workflows_pending_scope/ci.yml      .github/workflows/ci.yml
git mv .github/_workflows_pending_scope/release.yml .github/workflows/release.yml
git commit -m "ci: activate the CI/CD pipeline" .github/workflows/
git push
```

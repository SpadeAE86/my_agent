# GitHub Push Protection（GH013）最少修复步骤

适用场景：`git push` 被 GitHub 拦截，提示 `GH013: Repository rule violations` / `Push cannot contain secrets`。

## 结论先说

- `.gitignore` **只能忽略未被 Git 跟踪（untracked）** 的文件。
- 如果敏感文件 **已经被提交过**（tracked 且存在于某次 commit 中），即使你后来删除它，**Push Protection 仍会因为历史 commit 里出现过 secret 而拦截 push**。

因此修复分两类：

1) **仅“以后不再误提交”**：解除跟踪 + 保持本地文件（不涉及 push protection 历史扫描）。
2) **“这次 push 被拦了”**：必须让即将 push 的提交序列中 **不包含** 那个带 secret 的 commit（方法：丢弃本地脏提交并重新 commit；或重写历史；或 GitHub 解封链接放行）。

---

## A. 一劳永逸：解除跟踪（让 `.gitignore` 真正生效）

当你发现某些本地数据（如 `src/data/*.json`）总被带进 commit：

1. 确保 `.gitignore` 已写好规则（示例）：

```gitignore
src/data/**/*.json
**/data/**/*.json
```

2. 解除跟踪（保留本地文件）：

```bash
git rm -r --cached src/data
```

3. 提交并推送：

```bash
git commit -m "chore: stop tracking src/data json"
git push origin main
```

验证是否还在被跟踪：

```bash
git ls-files src/data
```

如果输出为空，说明 `src/data` 下已不再被 Git 跟踪，以后生成的 json 会被 `.gitignore` 忽略。

---

## B. 最少步骤修复 GH013（不重写远端历史，丢弃本地脏提交）

适用：**远端 `origin/main` 干净**，只有你本地多出来的一些 commit（其中包含 secret），导致 push 被拦。

目标：不改远端历史，只把你的有效改动“干净地”重新做成 1 个新提交再 push。

步骤：

1. 确认远端基线（并 fetch 最新）：

```bash
git fetch origin
git log --oneline --decorate -n 5
```

2. 从 `origin/main` 拉一条干净分支：

```bash
git switch -c clean-main origin/main
```

3. 把你“本地旧分支上的有效代码改动”搬到新分支（**不要带数据文件**）：

最稳方式：直接从旧分支 checkout 需要的路径（示例）：

```bash
# 假设你的旧分支叫 main（但它本地有脏 commit）
git checkout main -- src/infra src/models src/routers src/services src/utils src/test src/prompt_templates
```

4. 提交：

```bash
git commit -m "feat: <your changes>"
```

5. 推送到远端 main（不重写历史）：

```bash
git push origin clean-main:main
```

6. 本地 main 对齐远端（可选但推荐，避免继续在脏历史上工作）：

```bash
git switch main
git reset --hard origin/main
```

> 这种做法的优点：**不需要 force push**，不会改写 GitHub 上的 commit 历史，且能立刻解除 GH013。

---

## C. 什么时候需要重写历史？

如果**远端本身已经存在**包含 secret 的 commit（即远端也不干净），那么要么：

- 重写历史清除 secret（需要 force push，影响协作）
- 或使用 GitHub 提供的 unblock 链接放行（不推荐）

本仓库的优先策略：先采用 B（干净分支重做提交）避免重写历史。


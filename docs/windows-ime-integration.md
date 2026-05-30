# Windows 系统级输入法接入方案

> **文档状态**: 技术预研 / 尚未实现  
> **最后更新**: 2026-05-30

---

## 1. 当前状态

golf 目前是一个基于 **Tkinter** 的桌面应用原型，**不是** Windows 系统级输入法。

当前架构：

```
┌──────────────┐     ┌────────────────────┐
│  Tkinter GUI │────▶│  InputMethodEngine  │
│  (编辑器窗口) │◀────│  (Python 进程内)    │
└──────────────┘     └────────────────────┘
```

用户只能在 golf 自带的编辑器窗口中输入，无法在其他应用（如浏览器、Word、VS Code）中使用 golf 输入法。要实现系统级输入法，需要对接 Windows 的输入法框架。

---

## 2. TSF (Text Services Framework) 路线（推荐）

TSF 是 Windows Vista 以来的标准输入法框架，也是 Windows 10/11 唯一官方推荐的路线。

### 2.1 核心接口

需要实现的 COM 接口：

| 接口 | 用途 |
|------|------|
| `ITfTextInputProcessor` | 输入法生命周期入口 (Activate / Deactivate) |
| `ITfTextInputProcessorEx` | 扩展激活接口 (Windows 8+) |
| `ITfKeyEventSink` | 拦截按键事件 (OnKeyDown / OnKeyUp / OnTestKeyDown) |
| `ITfCompositionSink` | 管理组合字符串 (composing) 的创建、更新、结束 |
| `ITfCandidateListUIElement` | 提供候选词列表给系统绘制 |
| `ITfDisplayAttributeProvider` | 设置组合文字的下划线、颜色等显示属性 |
| `ITfThreadMgrEventSink` | 监听焦点变化（切换到不同文本控件时） |
| `ITfLangBarItemButton` | 语言栏图标和菜单 |

### 2.2 实现步骤

1. **创建 COM DLL**: 实现上述接口，编译为 `.dll`
2. **注册**: `regsvr32 golf_tsf.dll`
3. **GUID 注册**: 在注册表 `HKLM\SOFTWARE\Microsoft\CTF\TIP\{CLSID}` 下注册语言配置
4. **按键处理**: 在 `OnKeyDown` 中拦截字母键，构建 composing buffer
5. **候选窗定位**: 通过 `ITfContextView::GetTextExt` 获取光标矩形坐标
6. **候选词更新**: 调用 Python 引擎获取候选列表，通过 `ITfCandidateListUIElement` 展示

### 2.3 架构设计

```
┌─────────────────┐     Named Pipe / gRPC      ┌──────────────────────┐
│  TSF DLL (C++)  │◀───────────────────────────▶│  Python Engine       │
│  - COM 接口实现  │                             │  - PinyinGenerator   │
│  - 候选窗绘制    │     JSON / Protobuf         │  - FrequencyReranker │
│  - 按键拦截      │                             │  - 用户词库          │
└─────────────────┘                             └──────────────────────┘
        ▲
        │ COM
        ▼
┌─────────────────┐
│  Windows TSF    │
│  (系统输入管理)  │
└─────────────────┘
```

### 2.4 通信协议

TSF DLL 与 Python Engine 之间建议使用以下方式之一：

| 方案 | 延迟 | 复杂度 | 推荐场景 |
|------|------|--------|----------|
| **Named Pipe** | ~0.1ms | 低 | 首选，Windows 原生，实现简单 |
| **gRPC** | ~1-5ms | 中 | 需要跨平台或结构化 API 时 |
| **Shared Memory** | ~0.01ms | 高 | 极致性能要求 |
| **TCP Socket** | ~0.5ms | 低 | 最简实现，调试方便 |

建议初期使用 Named Pipe，消息格式为 JSON：

```json
// 请求
{"type": "query", "composing": "nihao", "context": "今天"}

// 响应
{"candidates": [{"text": "你好", "score": 9800.0}, {"text": "拟好", "score": 120.0}]}
```

---

## 3. IMM32 路线（已过时）

IMM32 (Input Method Manager) 是 Windows 95 时代的输入法接口，基于 `ImmInstallIME` 和 HKL 机制。

**不推荐使用的原因：**

- Windows 10/11 对 IMM32 的支持逐步削弱
- UWP / WinUI 3 应用不支持 IMM32
- 部分安全软件会拦截 IMM32 DLL 加载
- 微软官方已不再维护相关文档
- 新版 Windows 默认只加载 TSF 输入法

如果必须兼容极老版本 Windows (XP/7)，可考虑 IMM32，但 golf 项目不建议投入。

---

## 4. 模块边界

```
golf/
├── tsf/                          # TSF DLL 源码 (C++ 或 Rust)
│   ├── src/
│   │   ├── text_service.cpp      # ITfTextInputProcessor 实现
│   │   ├── key_event_sink.cpp    # 按键拦截
│   │   ├── composition.cpp       # 组合字符串管理
│   │   ├── candidate_ui.cpp      # 候选窗 UI
│   │   ├── pipe_client.cpp       # Named Pipe 客户端
│   │   └── register.cpp          # COM 注册
│   ├── CMakeLists.txt
│   └── golf_tsf.def
│
├── src/input_method/             # Python Engine (现有代码)
│   ├── engine.py                 # 核心引擎
│   ├── generator/                # 候选生成器
│   ├── reranker/                 # 排序器
│   └── pipe_server.py            # Named Pipe 服务端 (新增)
│
└── scripts/
    ├── install_tsf.ps1           # 注册脚本
    └── uninstall_tsf.ps1         # 卸载脚本
```

---

## 5. 构建工具链

### 5.1 C++ 路线

- **Visual Studio 2022** (Community Edition 即可)
- Windows SDK 10.0.22621.0+
- CMake 3.20+
- 关键头文件: `<msctf.h>`, `<ctffunc.h>`, `<inputscope.h>`

```cmake
# CMakeLists.txt 示例
cmake_minimum_required(VERSION 3.20)
project(golf_tsf LANGUAGES CXX)

add_library(golf_tsf SHARED
    src/text_service.cpp
    src/key_event_sink.cpp
    src/composition.cpp
    src/candidate_ui.cpp
    src/pipe_client.cpp
    src/register.cpp
)

target_link_libraries(golf_tsf PRIVATE ole32 oleaut32 uuid)
```

### 5.2 Rust 路线

- Rust stable (1.70+)
- `windows-rs` crate (官方 Windows API 绑定)
- 优势: 内存安全、无 UB 风险、更现代的构建系统

```toml
# Cargo.toml 示例
[dependencies]
windows = { version = "0.52", features = [
    "Win32_UI_TextServices",
    "Win32_System_Com",
    "Win32_Foundation",
    "implement",
]}
```

---

## 6. 注册 / 卸载步骤

### 6.1 注册

```powershell
# 以管理员权限运行
regsvr32 /s golf_tsf.dll

# 验证注册
Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\CTF\TIP\{YOUR-CLSID-HERE}"
```

### 6.2 卸载

```powershell
# 以管理员权限运行
regsvr32 /u /s golf_tsf.dll

# 清理注册表残留 (如有必要)
Remove-Item -Path "HKLM:\SOFTWARE\Microsoft\CTF\TIP\{YOUR-CLSID-HERE}" -Recurse
```

### 6.3 开发调试

开发期间建议：

1. 使用 `regsvr32` 注册 Debug 版本
2. 附加调试器到目标进程 (如 notepad.exe)
3. 在 `OnKeyDown` 设置断点
4. 使用 OutputDebugString 输出日志
5. 使用 [DebugView](https://learn.microsoft.com/en-us/sysinternals/downloads/debugview) 查看

---

## 7. 候选窗定位方案

### 7.1 系统提供的光标位置

```cpp
// 通过 TSF 获取光标位置
HRESULT hr;
ITfContextView* pView = nullptr;
hr = pContext->GetActiveView(&pView);

RECT rc;
BOOL fClipped;
hr = pView->GetTextExt(ecRead, pRangeComposition, &rc, &fClipped);

// rc 即为组合字符串在屏幕上的矩形坐标
// 候选窗应显示在 rc 的下方或上方
```

### 7.2 候选窗显示策略

- **首选**: 在光标下方显示候选窗
- **回退**: 如果下方空间不足 (接近屏幕底部)，改为上方显示
- **多显示器**: 使用 `MonitorFromRect` 确定正确的显示器
- **DPI 适配**: 使用 `GetDpiForWindow` 处理高 DPI 缩放

---

## 8. 参考项目

| 项目 | 语言 | 说明 |
|------|------|------|
| [PIME](https://github.com/nicedomain/PIME) | C++ + Python | TSF 框架 + Python 输入法后端，架构最接近 golf 的目标 |
| [RIME (小狼毫)](https://github.com/rime/weasel) | C++ | 成熟的开源输入法，TSF 实现可参考 |
| [Fcitx5](https://github.com/fcitx/fcitx5) | C++ | Linux 输入法框架，跨平台设计思路可借鉴 |
| [Windows TSF Samples](https://github.com/nicedomain/sample-ime) | C++ | 微软官方 TSF 示例代码 |
| [rust-tsf](https://github.com/nicedomain/rust-tsf) | Rust | Rust 实现的 TSF 输入法骨架 |

### 8.1 PIME 架构参考

PIME 的架构与 golf 的目标最为接近：

```
PIME 架构:
  TSF DLL (C++) ──Named Pipe──▶ PIMEServer (Python)
                                  ├── 酷音输入法模块
                                  ├── 新酷音模块
                                  └── 自定义输入法模块
```

golf 可以参考 PIME 的 pipe 通信协议和 DLL 注册流程，将 Python Engine 作为后端服务运行。

---

## 9. 实施路线建议

### Phase 1: 基础可用 (MVP)

- [ ] 选定实现语言 (C++ 或 Rust)
- [ ] 实现最小 TSF DLL: 能注册、能拦截按键
- [ ] 实现 Named Pipe 通信
- [ ] Python Engine 启动 pipe server
- [ ] 在 Notepad 中验证基本输入

### Phase 2: 功能完善

- [ ] 候选窗 UI (自绘或 Direct2D)
- [ ] 多候选翻页
- [ ] 组合字符串下划线显示
- [ ] 语言栏图标
- [ ] 模式切换 (中/英/日)

### Phase 3: 生产就绪

- [ ] 安装包 (NSIS / WiX)
- [ ] 自动更新机制
- [ ] 高 DPI / 多显示器适配
- [ ] 性能优化 (< 10ms 端到端延迟)
- [ ] 安全审计 (COM 权限、pipe 权限)

---

## 10. 注意事项

> **重要**: 系统级输入法开发涉及 COM 编程、内核态交互和全局按键拦截，复杂度远高于普通桌面应用。建议在 Tkinter 原型验证核心算法后，再投入 TSF 集成开发。

- TSF DLL 运行在每个使用输入法的进程中，任何崩溃都会导致目标应用崩溃
- COM 线程模型必须正确处理 (STA/MTA)
- 需要处理 UAC 权限 (注册需要管理员，使用不需要)
- 需要代码签名证书 (否则部分安全软件会拦截)

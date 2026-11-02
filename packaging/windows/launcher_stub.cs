using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace TensorFlowKIStudio
{
    internal sealed class SliceStream : Stream
    {
        private readonly Stream _baseStream;
        private readonly long _offset;
        private readonly long _length;
        private long _position;

        public SliceStream(Stream baseStream, long offset, long length)
        {
            _baseStream = baseStream;
            _offset = offset;
            _length = length;
            _position = 0;
            _baseStream.Position = _offset;
        }

        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return _length; } }

        public override long Position
        {
            get { return _position; }
            set { Seek(value, SeekOrigin.Begin); }
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            long remaining = _length - _position;
            if (remaining <= 0) return 0;
            if (count > remaining) count = (int)remaining;
            _baseStream.Position = _offset + _position;
            int read = _baseStream.Read(buffer, offset, count);
            _position += read;
            return read;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long target;
            switch (origin)
            {
                case SeekOrigin.Begin:
                    target = offset;
                    break;
                case SeekOrigin.Current:
                    target = _position + offset;
                    break;
                case SeekOrigin.End:
                    target = _length + offset;
                    break;
                default:
                    throw new ArgumentOutOfRangeException("origin");
            }
            if (target < 0) target = 0;
            if (target > _length) target = _length;
            _position = target;
            _baseStream.Position = _offset + _position;
            return _position;
        }

        public override void Flush() { }
        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }

    internal static class PayloadManager
    {
        // Replaced at build time by pack_single_exe.py
        public const string BUILD_ID = "__BUILD_ID_PLACEHOLDER__";
        private const string TRAILER_MAGIC = "TFKILITE";

        public static string GetAppDir()
        {
            string exePath = Application.ExecutablePath;
            return Path.GetDirectoryName(Path.GetFullPath(exePath));
        }

        public static bool IsRuntimeReady(string runtimeDir)
        {
            string appEntry = Path.Combine(runtimeDir, "run_studio.pyw");
            string mnistFile = Path.Combine(runtimeDir, "assets", "mnist.npz");
            string markerPath = Path.Combine(runtimeDir, "_engine_build_id.txt");
            if (!File.Exists(appEntry) || !File.Exists(mnistFile) || !File.Exists(markerPath))
                return false;
            try
            {
                string current = File.ReadAllText(markerPath, Encoding.UTF8).Trim();
                return string.Equals(current, BUILD_ID, StringComparison.Ordinal);
            }
            catch
            {
                return false;
            }
        }

        public static void ExtractPayload(string runtimeDir, Action<int, string, string> onProgress)
        {
            string exePath = Application.ExecutablePath;
            using (FileStream fs = new FileStream(exePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (fs.Length < 24)
                    throw new InvalidDataException("Ungültiges Anwendungs-Archiv (Trailer fehlt).");

                fs.Seek(-24, SeekOrigin.End);
                byte[] trailer = new byte[24];
                int n = 0;
                while (n < 24)
                {
                    int r = fs.Read(trailer, n, 24 - n);
                    if (r <= 0) break;
                    n += r;
                }

                long zipOffset = BitConverter.ToInt64(trailer, 0);
                long zipLength = BitConverter.ToInt64(trailer, 8);
                string magic = Encoding.ASCII.GetString(trailer, 16, 8);

                if (magic != TRAILER_MAGIC || zipOffset <= 0 || zipLength <= 0 || zipOffset + zipLength + 24 > fs.Length)
                    throw new InvalidDataException("Das eingebettete KI-Studio-Paket konnte nicht gelesen werden.");

                if (Directory.Exists(runtimeDir))
                {
                    try { Directory.Delete(runtimeDir, true); } catch { }
                }
                Directory.CreateDirectory(runtimeDir);
                string runtimeFull = Path.GetFullPath(runtimeDir);
                if (!runtimeFull.EndsWith(Path.DirectorySeparatorChar.ToString()))
                    runtimeFull += Path.DirectorySeparatorChar;

                using (SliceStream slice = new SliceStream(fs, zipOffset, zipLength))
                using (ZipArchive archive = new ZipArchive(slice, ZipArchiveMode.Read, true))
                {
                    long totalBytes = 0;
                    foreach (ZipArchiveEntry e in archive.Entries)
                    {
                        totalBytes += e.Length;
                    }
                    if (totalBytes <= 0) totalBytes = 1;

                    long doneBytes = 0;
                    byte[] buffer = new byte[131072];

                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        string rel = entry.FullName.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
                        string destPath = Path.GetFullPath(Path.Combine(runtimeDir, rel));
                        if (!destPath.StartsWith(runtimeFull, StringComparison.OrdinalIgnoreCase))
                            continue;

                        if (string.IsNullOrEmpty(entry.Name))
                        {
                            Directory.CreateDirectory(destPath);
                            continue;
                        }

                        string parentDir = Path.GetDirectoryName(destPath);
                        if (!string.IsNullOrEmpty(parentDir))
                            Directory.CreateDirectory(parentDir);

                        using (Stream inStream = entry.Open())
                        using (FileStream outStream = new FileStream(destPath, FileMode.Create, FileAccess.Write, FileShare.None, 131072))
                        {
                            int read;
                            while ((read = inStream.Read(buffer, 0, buffer.Length)) > 0)
                            {
                                outStream.Write(buffer, 0, read);
                                doneBytes += read;
                                int permille = (int)Math.Min(920L, (doneBytes * 880L) / totalBytes + 40L);
                                int pct = permille / 10;
                                if (onProgress != null)
                                {
                                    onProgress(
                                        permille,
                                        "Bereite KI-Modelle & Offline-MNIST vor...",
                                        pct.ToString() + " %"
                                    );
                                }
                            }
                        }
                    }
                }

                File.WriteAllText(Path.Combine(runtimeDir, "_engine_build_id.txt"), BUILD_ID + "\n", Encoding.UTF8);
            }
        }

        private static bool IsValidNonStubExe(string path)
        {
            if (string.IsNullOrEmpty(path)) return false;
            try
            {
                if (!File.Exists(path)) return false;
                FileInfo fi = new FileInfo(path);
                return fi.Length > 4096;
            }
            catch
            {
                return false;
            }
        }

        private static bool HasTensorFlowInstalled(string pythonExe)
        {
            try
            {
                string pyDir = Path.GetDirectoryName(Path.GetFullPath(pythonExe));
                if (string.IsNullOrEmpty(pyDir)) return false;

                if (Directory.Exists(Path.Combine(pyDir, "Lib", "site-packages", "tensorflow")))
                    return true;

                // Check user site-packages (%APPDATA%\Python\Python3xx\site-packages\tensorflow)
                string folderName = Path.GetFileName(pyDir);
                string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                if (!string.IsNullOrEmpty(appData) && !string.IsNullOrEmpty(folderName))
                {
                    if (Directory.Exists(Path.Combine(appData, "Python", folderName, "site-packages", "tensorflow")))
                        return true;
                }
            }
            catch { }
            return false;
        }

        private static void AddCandidate(List<string> list, HashSet<string> seen, string path)
        {
            if (!IsValidNonStubExe(path)) return;
            string full;
            try { full = Path.GetFullPath(path); } catch { return; }
            if (seen.Add(full.ToLowerInvariant()))
                list.Add(full);
        }

        private static void ScanPythonRegistry(RegistryKey root, List<string> list, HashSet<string> seen)
        {
            if (root == null) return;
            try
            {
                using (RegistryKey core = root.OpenSubKey(@"SOFTWARE\Python\PythonCore"))
                {
                    if (core == null) return;
                    string[] versions = core.GetSubKeyNames();
                    Array.Sort(versions, StringComparer.OrdinalIgnoreCase);
                    Array.Reverse(versions);
                    foreach (string ver in versions)
                    {
                        using (RegistryKey ip = core.OpenSubKey(ver + @"\InstallPath"))
                        {
                            if (ip == null) continue;
                            string exeVal = ip.GetValue("ExecutablePath") as string;
                            if (!string.IsNullOrEmpty(exeVal))
                                AddCandidate(list, seen, exeVal);
                            string dirVal = ip.GetValue("") as string;
                            if (!string.IsNullOrEmpty(dirVal))
                                AddCandidate(list, seen, Path.Combine(dirVal, "python.exe"));
                        }
                    }
                }
            }
            catch { }
        }

        private static void ScanPythonDirs(string parentDir, string pattern, List<string> list, HashSet<string> seen)
        {
            if (string.IsNullOrEmpty(parentDir) || !Directory.Exists(parentDir)) return;
            try
            {
                string[] dirs = Directory.GetDirectories(parentDir, pattern);
                Array.Sort(dirs, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(dirs);
                foreach (string d in dirs)
                {
                    AddCandidate(list, seen, Path.Combine(d, "python.exe"));
                }
            }
            catch { }
        }

        public static bool FindPythonCommand(string runtimeDir, out string exePath, out string prefixArg)
        {
            exePath = null;
            prefixArg = "";

            string envOverride = Environment.GetEnvironmentVariable("KI_STUDIO_PYTHON");
            if (IsValidNonStubExe(envOverride))
            {
                exePath = envOverride;
                return true;
            }

            string cacheFile = Path.Combine(runtimeDir, "_python_path.txt");
            if (File.Exists(cacheFile))
            {
                try
                {
                    string cached = File.ReadAllText(cacheFile, Encoding.UTF8).Trim();
                    if (IsValidNonStubExe(cached) && HasTensorFlowInstalled(cached))
                    {
                        exePath = cached;
                        return true;
                    }
                }
                catch { }
            }

            List<string> candidates = new List<string>();
            HashSet<string> seen = new HashSet<string>();

            // 1. PATH entries (excluding Microsoft Store WindowsApps 0-byte stubs)
            string pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (string rawDir in pathEnv.Split(';'))
            {
                string d = rawDir.Trim().Trim('"');
                if (d.Length == 0) continue;
                if (d.IndexOf(@"\Microsoft\WindowsApps", StringComparison.OrdinalIgnoreCase) >= 0) continue;
                try { AddCandidate(candidates, seen, Path.Combine(d, "python.exe")); } catch { }
            }

            // 2. Windows Registry & Common Python Install Directories
            ScanPythonRegistry(Registry.CurrentUser, candidates, seen);
            ScanPythonRegistry(Registry.LocalMachine, candidates, seen);

            string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrEmpty(localApp))
                ScanPythonDirs(Path.Combine(localApp, "Programs", "Python"), "Python*", candidates, seen);

            ScanPythonDirs(@"C:\", "Python*", candidates, seen);
            ScanPythonDirs(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Python*", candidates, seen);

            // Prefer a Python installation that already has TensorFlow installed!
            foreach (string cand in candidates)
            {
                if (HasTensorFlowInstalled(cand))
                {
                    exePath = cand;
                    try { File.WriteAllText(cacheFile, cand + "\n", Encoding.UTF8); } catch { }
                    return true;
                }
            }

            if (candidates.Count > 0)
            {
                exePath = candidates[0];
                return true;
            }

            // 3. Fallback: py.exe launcher
            string pyLauncher = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "py.exe");
            if (IsValidNonStubExe(pyLauncher))
            {
                exePath = pyLauncher;
                prefixArg = "-3";
                return true;
            }

            return false;
        }

        public static string BuildPythonArguments(string prefixArg, string scriptPath, string[] args)
        {
            StringBuilder sb = new StringBuilder();
            if (!string.IsNullOrEmpty(prefixArg))
            {
                sb.Append(prefixArg).Append(' ');
            }
            sb.Append('"').Append(scriptPath).Append('"');

            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string a = args[i];
                    if (a == "--test-splash-only") continue;
                    sb.Append(' ');
                    if (a.Length == 0 || a.IndexOf(' ') >= 0 || a.IndexOf('"') >= 0)
                    {
                        sb.Append('"').Append(a.Replace("\"", "\\\"")).Append('"');
                    }
                    else
                    {
                        sb.Append(a);
                    }
                }
            }
            return sb.ToString();
        }

        public static void CleanupDataFolderIfRequested(string dataDir)
        {
            string flagFile = Path.Combine(dataDir, "_delete_all_on_exit.flag");
            if (!File.Exists(flagFile)) return;

            for (int attempt = 0; attempt < 30; attempt++)
            {
                try
                {
                    if (Directory.Exists(dataDir))
                        Directory.Delete(dataDir, true);
                    if (!Directory.Exists(dataDir))
                        break;
                }
                catch
                {
                    Thread.Sleep(120);
                }
            }
        }
    }

    internal static class TaskbarIconApplier
    {
        private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        [DllImport("user32.dll")]
        private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

        [DllImport("user32.dll")]
        private static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetWindowTextLength(IntPtr hWnd);

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr SendMessageTimeout(
            IntPtr hWnd,
            uint Msg,
            IntPtr wParam,
            IntPtr lParam,
            uint fuFlags,
            uint uTimeout,
            out IntPtr lpdwResult
        );

        private const uint WM_SETICON = 0x0080;
        private const uint SMTO_ABORTIFHUNG = 0x0002;

        public static void ApplyIconToOpenAppWindow(Icon bigIcon, Icon smallIcon, IntPtr excludeHwnd)
        {
            if (bigIcon == null && smallIcon == null) return;
            IntPtr hBig = bigIcon != null ? bigIcon.Handle : IntPtr.Zero;
            IntPtr hSmall = smallIcon != null ? smallIcon.Handle : IntPtr.Zero;

            int[] delaysMs = new int[] { 80, 220, 500, 1000, 1800, 3000 };
            for (int d = 0; d < delaysMs.Length; d++)
            {
                Thread.Sleep(delaysMs[d]);
                try
                {
                    EnumWindows(delegate(IntPtr hWnd, IntPtr lParam)
                    {
                        if (hWnd == excludeHwnd || !IsWindowVisible(hWnd))
                            return true;
                        int len = GetWindowTextLength(hWnd);
                        if (len <= 0) return true;
                        StringBuilder sb = new StringBuilder(len + 1);
                        GetWindowText(hWnd, sb, sb.Capacity);
                        string title = sb.ToString();
                        if (title.IndexOf("TensorFlow KI-Studio", StringComparison.Ordinal) >= 0)
                        {
                            IntPtr dummy;
                            if (hBig != IntPtr.Zero)
                                SendMessageTimeout(hWnd, WM_SETICON, new IntPtr(1), hBig, SMTO_ABORTIFHUNG, 200, out dummy);
                            if (hSmall != IntPtr.Zero)
                                SendMessageTimeout(hWnd, WM_SETICON, new IntPtr(0), hSmall, SMTO_ABORTIFHUNG, 200, out dummy);
                        }
                        return true;
                    }, IntPtr.Zero);
                }
                catch { }
            }
        }
    }

    internal sealed class SplashForm : Form
    {
        [DllImport("user32.dll")]
        private static extern bool ReleaseCapture();

        [DllImport("user32.dll")]
        private static extern IntPtr SendMessage(IntPtr hWnd, int Msg, int wParam, int lParam);

        private readonly Bitmap _splashImage;
        private readonly Icon _bigAppIcon;
        private readonly Icon _smallAppIcon;
        private readonly System.Windows.Forms.Timer _animTimer;
        private readonly string[] _args;
        private readonly bool _testSplashOnly;

        private int _targetPermille = 120;
        private double _displayPermille = 50.0;
        private double _shimmerPhase = 0.0;
        private volatile string _statusText = "Starte TensorFlow KI-Studio...";
        private volatile string _rightText = "Sofort-Start";
        private bool _isHiddenForChild = false;

        public SplashForm(string[] args, bool testSplashOnly)
        {
            _args = args;
            _testSplashOnly = testSplashOnly;

            AutoScaleMode = AutoScaleMode.None;
            FormBorderStyle = FormBorderStyle.None;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(520, 270);
            BackColor = Color.FromArgb(6, 11, 22);
            DoubleBuffered = true;
            ShowInTaskbar = true;
            Text = "TensorFlow KI-Studio";

            Assembly asm = Assembly.GetExecutingAssembly();
            byte[] iconBytes = null;
            using (Stream iconStream = asm.GetManifestResourceStream("app_icon.ico"))
            {
                if (iconStream != null)
                {
                    using (MemoryStream ms = new MemoryStream())
                    {
                        iconStream.CopyTo(ms);
                        iconBytes = ms.ToArray();
                    }
                }
            }
            if (iconBytes != null)
            {
                using (MemoryStream ms1 = new MemoryStream(iconBytes))
                    _bigAppIcon = new Icon(ms1, new Size(256, 256));
                using (MemoryStream ms2 = new MemoryStream(iconBytes))
                    _smallAppIcon = new Icon(ms2, new Size(32, 32));
                Icon = _bigAppIcon;
            }

            using (Stream imgStream = asm.GetManifestResourceStream("splash.png"))
            {
                if (imgStream != null)
                    _splashImage = new Bitmap(imgStream);
            }

            _animTimer = new System.Windows.Forms.Timer();
            _animTimer.Interval = 30;
            _animTimer.Tick += OnAnimTick;
        }

        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                cp.ClassStyle |= 0x00020000; // CS_DROPSHADOW
                return cp;
            }
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            {
                ReleaseCapture();
                SendMessage(Handle, 0xA1, 0x2, 0);
            }
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            _animTimer.Start();

            Thread worker = new Thread(RunBootWorker);
            worker.IsBackground = true;
            worker.Start();
        }

        private void UpdateProgress(int permille, string status, string rightLabel)
        {
            Interlocked.Exchange(ref _targetPermille, permille);
            _statusText = status;
            _rightText = rightLabel;
        }

        private void OnAnimTick(object sender, EventArgs e)
        {
            if (_isHiddenForChild) return;

            int target = Interlocked.CompareExchange(ref _targetPermille, 0, 0);
            if (_displayPermille < target)
            {
                double diff = target - _displayPermille;
                _displayPermille += Math.Max(5.0, diff * 0.30);
                if (_displayPermille > target) _displayPermille = target;
            }
            else if (target >= 850 && target < 995 && _displayPermille < 965)
            {
                _displayPermille += 2.5;
            }

            _shimmerPhase += 0.055;
            if (_shimmerPhase > 1.0) _shimmerPhase -= 1.0;

            Invalidate(new Rectangle(0, 214, 520, 56));
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

            if (_splashImage != null)
                g.DrawImage(_splashImage, 0, 0, 520, 270);
            else
                g.Clear(Color.FromArgb(6, 11, 22));

            using (Font statusFont = new Font("Segoe UI", 9.25f, FontStyle.Bold))
            using (Font rightFont = new Font("Consolas", 9.25f, FontStyle.Bold))
            using (Brush textBrush = new SolidBrush(Color.FromArgb(232, 240, 252)))
            using (Brush cyanBrush = new SolidBrush(Color.FromArgb(96, 205, 255)))
            {
                g.DrawString(_statusText, statusFont, textBrush, new PointF(18f, 222f));
                string rt = _rightText ?? "";
                SizeF rtSize = g.MeasureString(rt, rightFont);
                g.DrawString(rt, rightFont, cyanBrush, new PointF(502f - rtSize.Width, 223f));
            }

            Rectangle trackRect = new Rectangle(18, 246, 484, 10);
            using (Brush trackBg = new SolidBrush(Color.FromArgb(3, 8, 18)))
            {
                g.FillRectangle(trackBg, trackRect);
            }

            int fillW = Math.Max(10, Math.Min(484, (int)Math.Round(484.0 * (_displayPermille / 1000.0))));
            Rectangle fillRect = new Rectangle(18, 246, fillW, 10);
            using (LinearGradientBrush grad = new LinearGradientBrush(
                new Rectangle(18, 246, 484, 10),
                Color.FromArgb(56, 189, 248),
                Color.FromArgb(16, 185, 129),
                LinearGradientMode.Horizontal))
            {
                g.FillRectangle(grad, fillRect);
            }

            int shimmerCenter = 18 + (int)(_shimmerPhase * (fillW + 80)) - 40;
            Rectangle shimmerRect = Rectangle.Intersect(fillRect, new Rectangle(shimmerCenter - 35, 246, 70, 10));
            if (shimmerRect.Width > 2)
            {
                using (LinearGradientBrush sb = new LinearGradientBrush(
                    new Rectangle(shimmerCenter - 36, 246, 72, 10),
                    Color.FromArgb(0, 255, 255, 255),
                    Color.FromArgb(115, 255, 255, 255),
                    LinearGradientMode.Horizontal))
                {
                    ColorBlend cb = new ColorBlend(3);
                    cb.Colors = new Color[]
                    {
                        Color.FromArgb(0, 255, 255, 255),
                        Color.FromArgb(120, 255, 255, 255),
                        Color.FromArgb(0, 255, 255, 255)
                    };
                    cb.Positions = new float[] { 0f, 0.5f, 1f };
                    sb.InterpolationColors = cb;
                    g.FillRectangle(sb, shimmerRect);
                }
            }

            using (Pen trackBorder = new Pen(Color.FromArgb(95, 56, 189, 248), 1f))
                g.DrawRectangle(trackBorder, trackRect);

            using (Pen cardBorder = new Pen(Color.FromArgb(115, 56, 189, 248), 1f))
                g.DrawRectangle(cardBorder, 0, 0, 519, 269);
        }

        private void RunBootWorker()
        {
            Stopwatch sw = Stopwatch.StartNew();
            string appDir = PayloadManager.GetAppDir();
            string dataDir = Path.Combine(appDir, "KI_Studio_Daten");
            string runtimeDir = Path.Combine(dataDir, "Laufzeit");
            IntPtr ownHwnd = IntPtr.Zero;
            try { Invoke((MethodInvoker)delegate { ownHwnd = Handle; }); } catch { }

            try
            {
                if (_testSplashOnly)
                {
                    UpdateProgress(450, "Teste Sofort-Ladebildschirm & Icon...", "45 %");
                    Thread.Sleep(180);
                    UpdateProgress(1000, "Ladebildschirm bereit!", "100 %");
                    Thread.Sleep(180);
                    SafeCloseApp();
                    return;
                }

                string staleFlag = Path.Combine(dataDir, "_delete_all_on_exit.flag");
                if (File.Exists(staleFlag))
                {
                    try { File.Delete(staleFlag); } catch { }
                }

                if (!PayloadManager.IsRuntimeReady(runtimeDir))
                {
                    UpdateProgress(40, "Bereite KI-Modelle & Offline-MNIST vor...", "4 %");
                    PayloadManager.ExtractPayload(runtimeDir, UpdateProgress);
                }

                string pyExe, pyPrefix;
                if (!PayloadManager.FindPythonCommand(runtimeDir, out pyExe, out pyPrefix))
                {
                    throw new FileNotFoundException(
                        "Es wurde keine Python-Installation auf diesem PC gefunden.\n\n" +
                        "Bitte installiere Python (python.org) und führe einmalig aus:\n" +
                        "    pip install tensorflow"
                    );
                }

                UpdateProgress(900, "Öffne TensorFlow KI-Studio Fenster...", "Start...");

                string appScript = Path.Combine(runtimeDir, "run_studio.pyw");
                string evtName = "Local\\TFKIReady_" + Guid.NewGuid().ToString("N");

                using (EventWaitHandle readyEvt = new EventWaitHandle(false, EventResetMode.ManualReset, evtName))
                {
                    ProcessStartInfo psi = new ProcessStartInfo();
                    psi.FileName = pyExe;
                    psi.Arguments = PayloadManager.BuildPythonArguments(pyPrefix, appScript, _args);
                    psi.WorkingDirectory = appDir;
                    psi.UseShellExecute = false;
                    psi.CreateNoWindow = true;
                    psi.EnvironmentVariables["KI_STUDIO_APP_DIR"] = appDir;
                    psi.EnvironmentVariables["KI_STUDIO_BUNDLE_DIR"] = runtimeDir;
                    psi.EnvironmentVariables["KI_STUDIO_IS_EXE"] = "1";
                    psi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
                    psi.EnvironmentVariables["PYTHONUTF8"] = "1";
                    psi.EnvironmentVariables["KI_STUDIO_READY_EVENT"] = evtName;

                    using (Process child = Process.Start(psi))
                    {
                        readyEvt.WaitOne(8000);
                        UpdateProgress(1000, "TensorFlow KI-Studio wird geöffnet...", "100 %");

                        Thread iconThread = new Thread(delegate()
                        {
                            TaskbarIconApplier.ApplyIconToOpenAppWindow(_bigAppIcon, _smallAppIcon, ownHwnd);
                        });
                        iconThread.IsBackground = true;
                        iconThread.Start();

                        long elapsed = sw.ElapsedMilliseconds;
                        if (elapsed < 400)
                            Thread.Sleep((int)(400 - elapsed));

                        HideSplashForRunningChild();

                        if (child != null)
                        {
                            child.WaitForExit();
                        }
                    }
                }

                PayloadManager.CleanupDataFolderIfRequested(dataDir);
                SafeCloseApp();
            }
            catch (Exception ex)
            {
                try
                {
                    BeginInvoke((MethodInvoker)delegate
                    {
                        _animTimer.Stop();
                        MessageBox.Show(
                            this,
                            "Hinweis beim Starten von TensorFlow KI-Studio:\n\n" + ex.Message,
                            "TensorFlow KI-Studio",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Warning
                        );
                        Application.Exit();
                    });
                }
                catch
                {
                    Environment.Exit(1);
                }
            }
        }

        private void HideSplashForRunningChild()
        {
            try
            {
                BeginInvoke((MethodInvoker)delegate
                {
                    _isHiddenForChild = true;
                    _animTimer.Stop();
                    Opacity = 0;
                    ShowInTaskbar = false;
                    Hide();
                });
            }
            catch { }
        }

        private void SafeCloseApp()
        {
            try
            {
                BeginInvoke((MethodInvoker)delegate
                {
                    _animTimer.Stop();
                    if (_splashImage != null) _splashImage.Dispose();
                    if (_bigAppIcon != null) _bigAppIcon.Dispose();
                    if (_smallAppIcon != null) _smallAppIcon.Dispose();
                    Application.Exit();
                });
            }
            catch
            {
                Environment.Exit(0);
            }
        }
    }

    internal static class Program
    {
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        private static extern int SetCurrentProcessExplicitAppUserModelID(string AppID);

        [STAThread]
        private static int Main(string[] args)
        {
            try { SetCurrentProcessExplicitAppUserModelID("TensorFlow.KIStudio.DesktopApp"); } catch { }

            bool headless = false;
            bool testSplash = false;
            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    if (args[i] == "--no-window") headless = true;
                    if (args[i] == "--test-splash-only") testSplash = true;
                }
            }

            if (headless && !testSplash)
            {
                return RunHeadless(args);
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SplashForm(args, testSplash));
            return 0;
        }

        private static int RunHeadless(string[] args)
        {
            string appDir = PayloadManager.GetAppDir();
            string dataDir = Path.Combine(appDir, "KI_Studio_Daten");
            string runtimeDir = Path.Combine(dataDir, "Laufzeit");

            string staleFlag = Path.Combine(dataDir, "_delete_all_on_exit.flag");
            if (File.Exists(staleFlag))
            {
                try { File.Delete(staleFlag); } catch { }
            }

            if (!PayloadManager.IsRuntimeReady(runtimeDir))
            {
                PayloadManager.ExtractPayload(runtimeDir, null);
            }

            string pyExe, pyPrefix;
            if (!PayloadManager.FindPythonCommand(runtimeDir, out pyExe, out pyPrefix))
            {
                Console.Error.WriteLine("FEHLER: Python wurde auf diesem PC nicht gefunden.");
                return 2;
            }

            string appScript = Path.Combine(runtimeDir, "run_studio.pyw");
            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = pyExe;
            psi.Arguments = PayloadManager.BuildPythonArguments(pyPrefix, appScript, args);
            psi.WorkingDirectory = appDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.EnvironmentVariables["KI_STUDIO_APP_DIR"] = appDir;
            psi.EnvironmentVariables["KI_STUDIO_BUNDLE_DIR"] = runtimeDir;
            psi.EnvironmentVariables["KI_STUDIO_IS_EXE"] = "1";
            psi.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";

            int exitCode = 0;
            using (Process child = Process.Start(psi))
            {
                if (child != null)
                {
                    child.OutputDataReceived += delegate(object s, DataReceivedEventArgs e)
                    {
                        if (e.Data != null)
                        {
                            try { Console.Out.WriteLine(e.Data); Console.Out.Flush(); } catch { }
                        }
                    };
                    child.ErrorDataReceived += delegate(object s, DataReceivedEventArgs e)
                    {
                        if (e.Data != null)
                        {
                            try { Console.Error.WriteLine(e.Data); Console.Error.Flush(); } catch { }
                        }
                    };
                    child.BeginOutputReadLine();
                    child.BeginErrorReadLine();
                    child.WaitForExit();
                    exitCode = child.ExitCode;
                }
            }

            PayloadManager.CleanupDataFolderIfRequested(dataDir);
            return exitCode;
        }
    }
}

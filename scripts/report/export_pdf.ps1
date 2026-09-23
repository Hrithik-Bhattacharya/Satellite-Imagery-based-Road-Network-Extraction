# Fills in the report's table of contents and page numbers with Microsoft Word, saves the
# .docx, and exports the PDF next to it.
#
# Usage (repo root, after build_report.py):
#   powershell -ExecutionPolicy Bypass -File scripts/report/export_pdf.ps1
param(
  [string]$Docx = "$PSScriptRoot\..\..\docs\report\Internship_Report.docx",
  [string]$Pdf  = "$PSScriptRoot\..\..\docs\report\Internship_Report.pdf"
)
$Docx = (Resolve-Path $Docx).Path
$Pdf  = [System.IO.Path]::GetFullPath($Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($Docx, $false, $false)
  $doc.Repaginate()
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
  $doc.Fields.Update() | Out-Null
  $doc.Repaginate()
  foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
  $pages = $doc.ComputeStatistics(2)
  $doc.Save()
  $doc.SaveAs([ref]$Pdf, [ref]17)
  $doc.Close([ref]0)
  Write-Output "pages=$pages"
  Write-Output "pdf=$Pdf"
} finally { $word.Quit() }

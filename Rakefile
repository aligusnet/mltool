desc 'Generate docs using Haddock and publish to GitHub Pages.'
task :publish_docs do
  # if this is a pull request, do a simple build of the site and stop
  if ENV['TRAVIS_PULL_REQUEST'].to_s.to_i > 0
    puts 'Pull request detected. No documentation needed'
    next
  end

  if ENV['PUBLISH_DOCS'] != 'true'
    puts 'Publishing is disabled for this build.'
    next
  end

  repo_dir = 'mltool-docs'
  repo_url = 'https://github.com/Alexander-Ignatyev/mltool-docs.git'
  haddock_path = `stack #{ENV['ARGS']} path --local-doc-root`
  haddock_path.strip!
  last_commit_message = `git log -1 --pretty=%B`
  last_commit_message.strip!

  system "[ -d '#{repo_dir}' ] && rm -rf '#{repo_dir}'"
  system "git clone #{repo_url} #{repo_dir} --depth 1"

  Dir.chdir(repo_dir)
  
  system "git config user.name '#{ENV['GIT_NAME']}'"
  system "git config user.email '#{ENV['GIT_EMAIL']}'"
  system 'git config credential.helper "store --file=.git/credentials"'

  system 'rm -rf doc'
  system "cp -r '#{haddock_path}' ."
  system 'git add . --all'
  system "git commit -m 'update-docs: #{last_commit_message}'"

  begin
    File.open('.git/credentials', 'w') do |f|
      f.write("https://#{ENV['GIT_TOKEN']}:x-oauth-basic@github.com")
    end
    system 'git push origin master'
  ensure
    File.delete '.git/credentials'
  end
end
